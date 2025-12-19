import cv2
cv2.setNumThreads(0)  # Fix segfault with GStreamer
import numpy as np
import time
import threading
from queue import Queue
import gi
import os
import sys

# Add parent directory to path to access utils and core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import create_camera_structure, calculate_display_dimensions, create_gstreamer_pipeline

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from utils.database_manager import DatabaseManager
from utils.similarity_search import SimilaritySearch
from utils.face_pipeline import FacePipeline


class FaceRTSPViewer:
    """
    Simple Face Recognition Viewer for RTSP streams.

    Displays real-time face detection and recognition results from RTSP cameras.
    """

    def __init__(self, stream_configs, similarity_threshold=0.6, detection_interval=3):
        """
        Initialize Face RTSP Viewer.

        Args:
            stream_configs (list): List of camera configuration dictionaries
            similarity_threshold (float): Face matching threshold (default: 0.6)
            detection_interval (int): Process every Nth frame (default: 3)
        """
        if len(stream_configs) != 2:
            raise ValueError("This system is designed for exactly 2 cameras")

        self.stream_configs = stream_configs
        self.similarity_threshold = similarity_threshold
        self.detection_interval = detection_interval

        # Initialize database for face recognition
        print("Initializing database manager...")
        self.db_manager = DatabaseManager()
        self.similarity_search = SimilaritySearch(self.db_manager)

        # GPU model will be initialized in worker thread
        self.face_pipeline = None

        # Initialize GStreamer
        Gst.init(None)

        # Display configuration
        self.camera_width = 640
        self.camera_height = 480
        self.display_width = 2 * self.camera_width
        self.display_height = self.camera_height

        # Initialize cameras
        self.cameras = {
            'center': create_camera_structure(stream_configs[0]),
            'right': create_camera_structure(stream_configs[1])
        }

        # Control flags
        self.running = False

        # Frame synchronization
        self.display_lock = threading.Lock()
        self.stable_display_buffer = {
            'center': None,
            'right': None,
            'timestamp': time.time()
        }

        # GPU job queue
        self.job_q = Queue(maxsize=64)

        # Stats
        self._stats = {'face_qwait_ms': [], 'face_infer_ms': []}
        self.gpu_worker_thread = None
        self.debug_counters = {'face_detections': 0, 'recognized_faces': 0, 'unknown_faces': 0}
        self.fps_counters = {'center': 0, 'right': 0}
        self.fps_start_times = {'center': time.time(), 'right': time.time()}
        self.current_fps = {'center': 0, 'right': 0}

        # Initialize pipelines
        self.initialize_cameras()
        print("FaceRTSPViewer initialized")

    def gpu_worker_loop(self):
        """GPU worker thread for face detection and recognition."""
        print("[GPU] Initializing FacePipeline...")
        self.face_pipeline = FacePipeline()

        # Warmup
        try:
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.face_pipeline.process_frame(dummy_frame, thresh=0.5, input_size=(640, 640))
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[GPU] Warmup note: {e}")

        print("[GPU] Ready for face recognition")

        while self.running:
            try:
                job = self.job_q.get(timeout=0.05)
            except:
                continue

            q_wait_ms = (time.time() - job['ts']) * 1000.0

            # Detect faces
            t0 = time.time()
            face_results = self.face_pipeline.process_frame(job['frame'], thresh=0.5, input_size=(640, 640))
            infer_ms = (time.time() - t0) * 1000.0
            self._stats['face_qwait_ms'].append(q_wait_ms)
            self._stats['face_infer_ms'].append(infer_ms)

            # Process recognized faces
            recognized_faces = []
            self.debug_counters['face_detections'] += len(face_results)

            for face_data in face_results:
                if face_data.get('embedding') is None:
                    continue

                match, sim = self.similarity_search.find_face_match_euclidean(
                    face_data['embedding'],
                    threshold=self.similarity_threshold
                )

                if match:
                    self.debug_counters['recognized_faces'] += 1
                    recognized_faces.append((face_data, match, sim))
                    print(f"[{job['camera_id']}] Recognized: {match['name']} (similarity: {sim:.3f})")
                else:
                    self.debug_counters['unknown_faces'] += 1
                    recognized_faces.append((face_data, None, sim))

            # Update camera detections
            camera_data = self.cameras[job['camera_id']]
            camera_data['persistent_detections'] = []

            for face_data, match, similarity in recognized_faces:
                detection = {
                    'bbox': face_data['bbox'],
                    'match': match,
                    'confidence': face_data['confidence'],
                    'similarity': similarity,
                    'type': 'face'
                }
                camera_data['persistent_detections'].append(detection)

            camera_data['detection_timestamp'] = time.time()

    def initialize_cameras(self):
        """Initialize camera pipelines."""
        for camera_id in ['center', 'right']:
            config = self.cameras[camera_id]['config']
            print(f"Initializing {camera_id} camera: {config.get('name', config['url'])}")
            try:
                self.create_camera_pipeline(camera_id, config)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error initializing {camera_id} camera: {e}")

    def create_camera_pipeline(self, camera_id, config):
        """Create GStreamer pipeline for camera."""
        try:
            pipeline, appsink = create_gstreamer_pipeline(
                camera_id, config, self.on_new_sample, self.on_bus_message
            )
            self.cameras[camera_id]['pipeline'] = pipeline
            self.cameras[camera_id]['appsink'] = appsink
        except Exception as e:
            print(f"Error creating pipeline for {camera_id}: {e}")
            raise

    def on_new_sample(self, appsink, camera_id):
        """Handle new frame from GStreamer."""
        try:
            sample = appsink.emit('pull-sample')
            if sample and self.running:
                buffer = sample.get_buffer()
                caps = sample.get_caps()

                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    structure = caps.get_structure(0)
                    width = structure.get_int('width')[1]
                    height = structure.get_int('height')[1]

                    frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                    frame = frame_data.reshape((height, width, 3)).copy()

                    with self.cameras[camera_id]['frame_lock']:
                        self.cameras[camera_id]['latest_raw_frame'] = frame
                        self.cameras[camera_id]['frame_counter'] += 1

                    frame_buffer = self.cameras[camera_id]['frame_buffer']
                    if frame_buffer.full():
                        try:
                            frame_buffer.get_nowait()
                        except:
                            pass
                    frame_buffer.put(frame)

                    buffer.unmap(map_info)

        except Exception as e:
            print(f"Error in frame capture for {camera_id}: {e}")

        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message, camera_id):
        """Handle GStreamer messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Pipeline error for {camera_id}: {err}")
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.cameras[camera_id]['pipeline']:
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    self.cameras[camera_id]['playing'] = True
                    print(f"Camera {camera_id} started")

        return True

    def processing_thread(self, camera_id):
        """Camera processing thread."""
        while self.running:
            try:
                camera_data = self.cameras[camera_id]
                frame_buffer = camera_data['frame_buffer']

                if not frame_buffer.empty():
                    frame = frame_buffer.get()

                    with camera_data['frame_lock']:
                        frame_count = camera_data['frame_counter']

                    should_detect = (frame_count % self.detection_interval == 0)

                    if should_detect:
                        try:
                            self.job_q.put_nowait({'camera_id': camera_id, 'frame': frame, 'ts': time.time()})
                        except:
                            pass

                    # Draw detections
                    current_time = time.time()
                    if (current_time - camera_data['detection_timestamp']) < 2.0:
                        processed_frame = self.draw_detections(frame, camera_data['persistent_detections'], camera_id)
                    else:
                        processed_frame = self.draw_detections(frame, [], camera_id)

                    with camera_data['frame_lock']:
                        camera_data['latest_processed_frame'] = processed_frame

                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error in processing thread for {camera_id}: {e}")
                time.sleep(0.1)

    def draw_detections(self, frame, detections, camera_id):
        """Draw face detections on frame."""
        camera_data = self.cameras[camera_id]
        new_width, new_height, pad_left, pad_top, pad_right, pad_bottom = \
            calculate_display_dimensions(frame.shape[1], frame.shape[0], self.camera_width, self.camera_height)

        resized_frame = cv2.resize(frame, (new_width, new_height))
        display_frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        display_frame[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = resized_frame

        scale_x = new_width / frame.shape[1]
        scale_y = new_height / frame.shape[0]

        # Draw face boxes
        for detection in detections:
            bbox = detection['bbox']
            match = detection['match']
            similarity = detection.get('similarity', 0.0)

            x1 = int(bbox[0] * scale_x) + pad_left
            y1 = int(bbox[1] * scale_y) + pad_top
            x2 = int(bbox[2] * scale_x) + pad_left
            y2 = int(bbox[3] * scale_y) + pad_top

            if match:
                person_name = match['name']
                color = (0, 255, 0)  # Green
                label = f"{person_name} ({similarity:.2f})"
            else:
                color = (0, 0, 255)  # Red
                label = f"Unknown ({similarity:.2f})"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Camera name
        camera_name = camera_data['config'].get('name', f'Camera {camera_id.title()}')
        cv2.putText(display_frame, camera_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # FPS
        fps_text = f"FPS: {self.current_fps[camera_id]}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display_frame

    def calculate_fps(self, camera_id):
        """Calculate FPS."""
        self.fps_counters[camera_id] += 1
        if time.time() - self.fps_start_times[camera_id] >= 1.0:
            self.current_fps[camera_id] = self.fps_counters[camera_id]
            self.fps_counters[camera_id] = 0
            self.fps_start_times[camera_id] = time.time()

    def update_stable_display(self):
        """Update display buffer."""
        with self.display_lock:
            updated = False
            for camera_id in ['center', 'right']:
                camera_data = self.cameras[camera_id]
                with camera_data['frame_lock']:
                    if camera_data['latest_processed_frame'] is not None:
                        self.stable_display_buffer[camera_id] = camera_data['latest_processed_frame'].copy()
                        updated = True
                        self.calculate_fps(camera_id)
            if updated:
                self.stable_display_buffer['timestamp'] = time.time()

    def create_stable_dual_display(self):
        """Create side-by-side display."""
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        with self.display_lock:
            # Center camera
            if self.stable_display_buffer['center'] is not None:
                canvas[0:self.camera_height, 0:self.camera_width] = self.stable_display_buffer['center']
            else:
                placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting Center Camera...", (50, self.camera_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                canvas[0:self.camera_height, 0:self.camera_width] = placeholder

            # Right camera
            if self.stable_display_buffer['right'] is not None:
                canvas[0:self.camera_height, self.camera_width:self.display_width] = self.stable_display_buffer['right']
            else:
                placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting Right Camera...", (50, self.camera_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                canvas[0:self.camera_height, self.camera_width:self.display_width] = placeholder

        self.draw_global_status(canvas)
        return canvas

    def draw_global_status(self, canvas):
        """Draw global statistics."""
        y_pos = 200
        line_height = 35

        cv2.putText(canvas, "Face Recognition Viewer", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height

        stats_text = f"Detected:{self.debug_counters['face_detections']} | Recognized:{self.debug_counters['recognized_faces']} | Unknown:{self.debug_counters['unknown_faces']}"
        cv2.putText(canvas, stats_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    def start(self):
        """Start the face recognition viewer."""
        # Start pipelines
        for camera_id in ['center', 'right']:
            try:
                pipeline = self.cameras[camera_id]['pipeline']
                ret = pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    print(f"Failed to start {camera_id} camera")
                time.sleep(1.0)
            except Exception as e:
                print(f"Error starting {camera_id}: {e}")

        time.sleep(3)
        self.running = True

        # Start GPU worker
        self.gpu_worker_thread = threading.Thread(target=self.gpu_worker_loop, daemon=True)
        self.gpu_worker_thread.start()

        # Start processing threads
        for camera_id in ['center', 'right']:
            thread = threading.Thread(target=self.processing_thread, args=(camera_id,), daemon=True)
            thread.start()
            self.cameras[camera_id]['thread'] = thread
            time.sleep(0.2)

        try:
            while self.running:
                self.update_stable_display()
                display = self.create_stable_dual_display()

                if not os.getenv('HEADLESS'):
                    cv2.imshow('Face Recognition Viewer', display)

                key = cv2.waitKey(33) & 0xFF if not os.getenv('HEADLESS') else -1
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Stop the system."""
        self.running = False

        for camera_id in ['center', 'right']:
            try:
                pipeline = self.cameras[camera_id]['pipeline']
                if pipeline:
                    pipeline.set_state(Gst.State.NULL)

                thread = self.cameras[camera_id].get('thread')
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
            except Exception as e:
                print(f"Error stopping {camera_id}: {e}")

        cv2.destroyAllWindows()
