#!/usr/bin/env python3
"""
Simple Face Detection and Recognition on RTSP Stream

This script:
1. Connects to a single RTSP camera using GStreamer
2. Detects faces in real-time using FacePipeline
3. Matches detected faces against stored face embeddings in database
4. Displays recognition results with bounding boxes and names

Usage:
    python face_detection_module/face_rtsp_recognition.py
"""

import cv2
cv2.setNumThreads(0)  # Fix segfault with GStreamer
import numpy as np
import time
import threading
from queue import Queue
import gi
import os
import sys

# Fix libproxy crash with RTSP URLs
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import create_gstreamer_pipeline
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from utils.database_manager import DatabaseManager
from utils.similarity_search import SimilaritySearch
from utils.face_pipeline import FacePipeline


class SimpleFaceRecognition:
    """Simple face recognition on single RTSP stream."""

    def __init__(self, rtsp_url, similarity_threshold=0.69, detection_interval=3):
        """
        Initialize face recognition viewer.

        Args:
            rtsp_url (str): RTSP camera URL
            similarity_threshold (float): Face matching threshold
            detection_interval (int): Process every Nth frame
        """
        self.rtsp_url = rtsp_url
        self.similarity_threshold = similarity_threshold
        self.detection_interval = detection_interval

        # Initialize database
        print("Initializing database...")
        self.db_manager = DatabaseManager()
        self.similarity_search = SimilaritySearch(self.db_manager)

        # Initialize FacePipeline (will load on GPU worker thread)
        self.face_pipeline = None

        # GStreamer
        Gst.init(None)
        self.pipeline = None
        self.appsink = None

        # Frame handling
        self.running = False
        self.frame_buffer = Queue(maxsize=2)
        self.frame_counter = 0
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Detection data
        self.detections = []
        self.detection_timestamp = 0
        self.detection_lock = threading.Lock()

        # GPU job queue
        self.job_q = Queue(maxsize=32)
        self.gpu_worker_thread = None

        # Stats
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0

        print("SimpleFaceRecognition initialized")

    def create_pipeline(self):
        """Create GStreamer pipeline."""
        camera_config = {
            'name': 'RTSP Camera',
            'url': self.rtsp_url,
            'width': 640,
            'height': 640
        }

        self.pipeline, self.appsink = create_gstreamer_pipeline(
            'camera', camera_config, self.on_new_sample, self.on_bus_message
        )

        print(f"GStreamer pipeline created for {self.rtsp_url}")

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

                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_counter += 1

                    # Add to processing buffer
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except:
                            pass
                    self.frame_buffer.put(frame)

                    buffer.unmap(map_info)

        except Exception as e:
            print(f"Error in frame capture: {e}")

        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message, camera_id):
        """Handle GStreamer messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Pipeline error: {err}")
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    print("Camera pipeline started")

        return True

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
                job = self.job_q.get(timeout=0.1)
            except:
                continue

            # Detect faces
            face_results = self.face_pipeline.process_frame(job['frame'], thresh=0.5, input_size=(640, 640))

            # Process recognized faces
            recognized_faces = []

            for face_data in face_results:
                if face_data.get('embedding') is None:
                    continue

                # Convert similarity threshold to distance threshold
                # similarity = 1.0 / (1.0 + distance)
                # Solving for distance: distance = (1.0 / similarity) - 1.0
                distance_threshold = (1.0 / self.similarity_threshold) - 1.0

                match, sim = self.similarity_search.find_face_match_euclidean(
                    face_data['embedding'],
                    threshold=distance_threshold
                )

                # Even if no match, get the best similarity from database
                if sim == 0.0 and match is None:
                    # Query for best match regardless of threshold
                    try:
                        conn = self.similarity_search.db_manager.get_connection()
                        if conn:
                            cur = conn.cursor()
                            vec = face_data['embedding'].astype(np.float32)
                            norm = np.linalg.norm(vec)
                            if norm > 0:
                                vec = vec / norm
                                cur.execute("""
                                    SELECT person_name, embedding <-> %s::vector AS distance
                                    FROM face_embeddings
                                    ORDER BY distance ASC
                                    LIMIT 1;
                                """, (vec.tolist(),))
                                row = cur.fetchone()
                                if row:
                                    _, distance = row
                                    sim = 1.0 / (1.0 + distance)
                            cur.close()
                            conn.close()
                    except Exception as e:
                        print(f"Error getting best similarity: {e}")

                if match:
                    recognized_faces.append((face_data, match, sim))
                    print(f"Recognized: {match['name']} (similarity: {sim:.3f})")
                else:
                    recognized_faces.append((face_data, None, sim))
                    print(f"Unknown face (best similarity: {sim:.3f})")

            # Update detections
            with self.detection_lock:
                self.detections = []
                for face_data, match, similarity in recognized_faces:
                    detection = {
                        'bbox': face_data['bbox'],
                        'match': match,
                        'confidence': face_data['confidence'],
                        'similarity': similarity
                    }
                    self.detections.append(detection)
                self.detection_timestamp = time.time()

    def processing_thread(self):
        """Frame processing thread."""
        while self.running:
            try:
                if not self.frame_buffer.empty():
                    frame = self.frame_buffer.get()

                    with self.frame_lock:
                        frame_count = self.frame_counter

                    # Send to GPU worker at interval
                    should_detect = (frame_count % self.detection_interval == 0)
                    if should_detect:
                        try:
                            self.job_q.put_nowait({'frame': frame, 'ts': time.time()})
                        except:
                            pass

                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(0.1)

    def draw_detections(self, frame):
        """Draw face detections on frame."""
        display_frame = frame.copy()

        with self.detection_lock:
            current_time = time.time()
            # Only show detections if recent (within 2 seconds)
            if (current_time - self.detection_timestamp) < 2.0:
                detections = self.detections.copy()
            else:
                detections = []

        # Draw face boxes
        for detection in detections:
            bbox = detection['bbox']
            match = detection['match']
            similarity = detection.get('similarity', 0.0)

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if match:
                person_name = match['name']
                color = (0, 255, 0)  # Green
                label = f"{person_name} ({similarity:.2f})"
            else:
                color = (0, 0, 255)  # Red
                label = f"Unknown ({similarity:.2f})"

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add FPS
        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Add info
        cv2.putText(display_frame, "Face Recognition Viewer", (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display_frame

    def calculate_fps(self):
        """Calculate FPS."""
        self.fps_counter += 1
        if time.time() - self.fps_start >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start = time.time()

    def start(self):
        """Start the face recognition viewer."""
        # Create and start pipeline
        self.create_pipeline()

        print("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start camera")
            return

        time.sleep(3)  # Wait for camera initialization
        self.running = True

        # Start GPU worker thread
        self.gpu_worker_thread = threading.Thread(target=self.gpu_worker_loop, daemon=True)
        self.gpu_worker_thread.start()

        # Start processing thread
        proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        proc_thread.start()

        print("Face recognition viewer started. Press 'q' to quit.")

        try:
            while self.running:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                    else:
                        frame = None

                if frame is not None:
                    display = self.draw_detections(frame)
                    cv2.imshow('Face Recognition', display)
                    self.calculate_fps()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Stop the system."""
        print("Stopping...")
        self.running = False

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        cv2.destroyAllWindows()
        print("Stopped")


def main():
    """Main entry point."""
    # RTSP camera URL
    RTSP_URL = 'rtsp://admin:admin%40123@192.168.0.216:554/stream1'
    SIMILARITY_THRESHOLD = 0.5
    DETECTION_INTERVAL = 3

    print("=" * 50)
    print("Face Recognition on RTSP Stream")
    print("=" * 50)
    print(f"Camera: {RTSP_URL}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Detection Interval: Every {DETECTION_INTERVAL} frames")
    print("=" * 50)

    # Change to parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)

    try:
        viewer = SimpleFaceRecognition(
            rtsp_url=RTSP_URL,
            similarity_threshold=SIMILARITY_THRESHOLD,
            detection_interval=DETECTION_INTERVAL
        )

        viewer.start()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
