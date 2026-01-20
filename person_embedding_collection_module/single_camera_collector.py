import cv2
cv2.setNumThreads(0)  # Fix segfault with GStreamer
import numpy as np
import time
import threading
from queue import Queue
import gi
import os
import sys
import torch
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path to access utils and core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import create_camera_structure, calculate_display_dimensions, create_gstreamer_pipeline

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from utils.database_manager import DatabaseManager
from utils.similarity_search import SimilaritySearch
from utils.face_pipeline import FacePipeline
from utils.person_detector import PersonDetector
from utils.person_embedder import PersonEmbedder
from utils.Live_gallery import LiveGallery

class Mode(Enum):
    FACES = 0
    COLLECT = 1

@dataclass
class Job:
    kind: str              # 'FACE' | 'PERSON'
    camera_id: str
    frame: np.ndarray
    ts: float


class SingleCameraCollector:
    """
    Single Camera Person Embedding Collector.

    Uses one camera for both face recognition and body embedding collection.
    """

    def __init__(self, camera_config, similarity_threshold=0.6, target_embeddings_per_person=25, output_dir=None):
        """
        Initialize Single Camera Collector.

        Args:
            camera_config (dict): Camera configuration (for both face recognition and body embeddings)
            similarity_threshold (float): Face matching threshold (default: 0.6)
            target_embeddings_per_person (int): Target embeddings per person (default: 25)
            output_dir (str): Directory to save .pt files (default: script_directory/embeddings_output)
        """
        self.similarity_threshold = similarity_threshold
        self.target_embeddings_per_person = target_embeddings_per_person
        self.live_gallery = LiveGallery(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set output directory
        if output_dir is None:
            repo_root = Path(__file__).parent.parent.absolute()
            self.output_dir = repo_root / 'embeddings_output'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Embeddings will be saved to: {self.output_dir}")

        # Initialize database
        print("Initializing database manager...")
        self.db_manager = DatabaseManager()

        # Initialize CPU-side utilities only
        print("Initializing similarity search...")
        self.similarity_search = SimilaritySearch(self.db_manager)

        # GPU models will be initialized in worker thread
        self.face_pipeline = None
        self.person_detector = None
        self.person_embedder = None

        # Initialize GStreamer
        Gst.init(None)

        # Display configuration (single camera)
        self.camera_width = 640
        self.camera_height = 480
        self.display_width = self.camera_width
        self.display_height = self.camera_height

        # Processing configuration
        self.detection_interval = 2  # Process every 2nd frame

        # Initialize single camera
        self.cameras = {
            'camera': create_camera_structure(camera_config)
        }

        # Store config for display
        self.cameras['camera']['config'] = camera_config

        # Control flags
        self.running = False
        self.collecting_embeddings = True

        # Frame synchronization
        self.display_lock = threading.Lock()
        self.stable_display_buffer = {
            'camera': None,
            'timestamp': time.time()
        }

        # Active collection tracking
        self.collection_lock = threading.Lock()
        self.active_collection = None  # Only one person at a time

        # Inference mode (mutually exclusive)
        self.mode = Mode.FACES
        self.mode_lock = threading.Lock()

        # Bounded GPU job queue
        self.job_q = Queue(maxsize=64)

        # For worker metrics
        self._stats = {
            'face_qwait_ms': [], 'face_infer_ms': [],
            'person_qwait_ms': [], 'person_infer_ms': []
        }

        # Worker thread handle
        self.gpu_worker_thread = None

        # Debug counters
        self.debug_counters = {
            'face_detections': 0,
            'recognized_faces': 0,
            'person_detections': 0,
            'embedding_attempts': 0,
            'embedding_successes': 0
        }

        # Designated zone for person positioning (normalized coordinates for 640x480)
        self.designated_zone = np.array([[478, 484], [757, 481], [802, 655], [439, 661]])

        # Person tracking for zone-based collection
        self.person_tracking_lock = threading.Lock()
        self.waiting_person = None  # {'name': str, 'face_id': int, 'entered_zone_time': float}
        self.countdown_duration = 2.0  # seconds to wait before starting collection

        # Performance tracking
        self.fps_counters = {'camera': 0}
        self.fps_start_times = {'camera': time.time()}
        self.current_fps = {'camera': 0}

        # Initialize pipelines
        self.initialize_cameras()

        print("SingleCameraCollector initialized - GPU models will load in worker thread")

    def update_embedding_status(self, person_id, person_name, pt_path, status='READY'):
        """Update the embedding_status table in the database."""
        try:
            current_timestamp = int(time.time() * 1000)

            conn = self.db_manager.get_connection()
            if not conn:
                print(f"[DB] Failed to get database connection for {person_name}")
                return False

            cur = conn.cursor()

            query = """
                INSERT INTO embedding_status (person_id, person_name, status, pt_path, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (person_id)
                DO UPDATE SET
                    person_name = EXCLUDED.person_name,
                    status = EXCLUDED.status,
                    pt_path = EXCLUDED.pt_path,
                    updated_at = EXCLUDED.updated_at
            """

            cur.execute(query, (person_id, person_name, status, str(pt_path), current_timestamp, current_timestamp))

            conn.commit()
            cur.close()
            conn.close()

            print(f"[DB] Updated embedding_status for {person_name} (person_id={person_id}) with status={status}")
            return True

        except Exception as e:
            print(f"[DB] Error updating embedding_status for {person_name}: {e}")
            return False

    def get_embedding_status(self, person_id):
        """
        Get the current status of a person's embeddings from the database.

        Args:
            person_id (int): Person ID from face_embeddings table

        Returns:
            str: Status ('READY', 'CONSUMED', 'PENDING', None if not found)
        """
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                return None

            cur = conn.cursor()
            cur.execute("SELECT status FROM embedding_status WHERE person_id = %s", (person_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                return row[0]
            return None

        except Exception as e:
            print(f"[DB] Error getting embedding_status: {e}")
            return None

    def save_person_embeddings_to_pt(self, person_name, embeddings_list, visibility_list, face_id):
        """Save collected embeddings to .pt file and set status to READY."""
        try:
            if not embeddings_list or not visibility_list:
                print(f"[PT] No embeddings to save for {person_name}")
                return False

            # Create person-specific directory
            person_dir = self.output_dir / person_name
            person_dir.mkdir(parents=True, exist_ok=True)

            # Concatenate all embeddings and visibility scores
            all_embeddings = torch.cat(embeddings_list, dim=0)
            all_visibility = torch.cat(visibility_list, dim=0)

            # Move to CPU if on GPU
            all_embeddings = all_embeddings.cpu() if all_embeddings.is_cuda else all_embeddings
            all_visibility = all_visibility.cpu() if all_visibility.is_cuda else all_visibility

            # Create PIDs tensor
            num_samples = all_embeddings.shape[0]
            pids = torch.tensor([face_id] * num_samples, dtype=torch.long)

            # Save embeddings (OVERWRITE existing files)
            torch.save(all_embeddings, person_dir / 'embeddings.pt')
            torch.save(all_visibility, person_dir / 'visibility.pt')
            torch.save(pids, person_dir / 'pids.pt')

            print(f"[PT] Saved {all_embeddings.shape[0]} embeddings for {person_name} (PID={face_id}) to {person_dir}")
            print(f"[PT]   - Embeddings shape: {all_embeddings.shape}")
            print(f"[PT]   - Visibility shape: {all_visibility.shape}")
            print(f"[PT]   - PIDs shape: {pids.shape}")

            # Update embedding_status table with READY status
            self.update_embedding_status(
                person_id=face_id,
                person_name=person_name,
                pt_path=person_dir,
                status='READY'
            )

            return True

        except Exception as e:
            print(f"[PT] Error saving embeddings for {person_name}: {e}")
            return False

    def should_start_collection(self, person_name, face_id):
        """
        Check if collection should start for a person based on embedding status.

        Collection is allowed when:
        - Status is None (never collected)
        - Status is 'CONSUMED' (person has been processed, can collect again)
        - Status is 'PENDING' (incomplete collection)

        Collection is blocked when:
        - Status is 'READY' (embeddings are complete and ready for gym system)

        Args:
            person_name (str): Name of the person
            face_id (int): Face ID from database

        Returns:
            bool: True if collection should start, False otherwise
        """
        status = self.get_embedding_status(face_id)

        if status == 'READY':
            print(f"[COLLECTION] {person_name} has READY embeddings - person can proceed to gym")
            return False
        elif status == 'CONSUMED':
            print(f"[COLLECTION] {person_name} status is CONSUMED - allowing re-collection")
            return True
        elif status == 'PENDING':
            print(f"[COLLECTION] {person_name} status is PENDING - re-collecting")
            return True
        elif status is None:
            print(f"[COLLECTION] {person_name} has no embeddings - starting fresh collection")
            return True
        else:
            print(f"[COLLECTION] {person_name} has unknown status '{status}' - allowing collection")
            return True

    def is_person_in_zone(self, bbox):
        """
        Check if person's feet (bottom-right corner of bbox) is inside the designated zone.

        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]

        Returns:
            bool: True if person's feet are in zone, False otherwise
        """
        # Use bottom-right corner (x2, y2) which represents the person's feet position
        feet_x = bbox[2]  # x_max (right side of bbox)
        feet_y = bbox[3]  # y_max (bottom of bbox, where feet are)

        # Use OpenCV point-in-polygon test
        result = cv2.pointPolygonTest(self.designated_zone.astype(np.float32), (float(feet_x), float(feet_y)), False)
        return result >= 0  # >= 0 means inside or on the boundary

    def gpu_worker_loop(self):
        """GPU worker thread that processes face detection and person embedding jobs."""
        # Initialize models
        print("[GPU] Initializing FacePipeline/PersonDetector/PersonEmbedder...")
        self.face_pipeline = FacePipeline()
        self.person_detector = PersonDetector()
        self.person_embedder = PersonEmbedder()

        # Warmup
        try:
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            _ = self.face_pipeline.process_frame(dummy_frame, thresh=0.5, input_size=(640, 640))
            _ = self.person_detector.detect_persons(dummy_frame)
            _, _ = self.person_embedder.extract_embedding(dummy_crop)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[GPU] Warmup note: {e}")

        print("[GPU] Ready.")

        while self.running:
            try:
                job = self.job_q.get(timeout=0.05)
            except:
                # Print metrics every ~5s
                if int(time.time()) % 5 == 0:
                    fq = np.mean(self._stats['face_qwait_ms'][-50:] or [0])
                    fi = np.mean(self._stats['face_infer_ms'][-50:] or [0])
                    pq = np.mean(self._stats['person_qwait_ms'][-50:] or [0])
                    pi = np.mean(self._stats['person_infer_ms'][-50:] or [0])
                    with self.mode_lock:
                        mode_name = self.mode.name
                    print(f"[GPU] face qwait={fq:.1f}ms infer={fi:.1f}ms | person qwait={pq:.1f}ms infer={pi:.1f}ms | mode={mode_name}")
                continue

            q_wait_ms = (time.time() - job.ts) * 1000.0
            with self.mode_lock:
                mode = self.mode

            if mode == Mode.FACES:
                # Process both FACE and PERSON jobs in FACES mode
                if job.kind == 'FACE' and job.camera_id == 'camera':
                    # Face detection and recognition
                    t0 = time.time()
                    face_results = self.face_pipeline.process_frame(job.frame, thresh=0.5, input_size=(640, 640))
                    infer_ms = (time.time() - t0) * 1000.0
                    self._stats['face_qwait_ms'].append(q_wait_ms)
                    self._stats['face_infer_ms'].append(infer_ms)

                    # Recognize faces
                    recognized_faces = []
                    self.debug_counters['face_detections'] += len(face_results)
                    print(f"[FACE] Detected {len(face_results)} faces in camera")

                    for face_data in face_results:
                        if face_data.get('embedding') is None:
                            print(f"[FACE] No embedding extracted for detected face in {job.camera_id}")
                            continue

                        # Convert similarity threshold to distance threshold
                        distance_threshold = (1.0 / self.similarity_threshold) - 1.0

                        match, sim = self.similarity_search.find_face_match_euclidean(
                            face_data['embedding'], threshold=distance_threshold
                        )
                        if match:
                            self.debug_counters['recognized_faces'] += 1
                            recognized_faces.append((face_data, match))
                            print(f"[FACE] Recognized: {match['name']} (similarity: {sim:.3f})")
                        else:
                            print(f"[FACE] Face detected but no match found (best similarity: {sim:.3f}, threshold: {self.similarity_threshold})")

                    # Update camera display with face detections
                    camera_data = self.cameras[job.camera_id]
                    camera_data['persistent_detections'] = []
                    for face_data, match in recognized_faces:
                        camera_data['persistent_detections'].append({
                            'bbox': face_data['bbox'],
                            'match': match,
                            'confidence': face_data['confidence'],
                            'type': 'face'
                        })
                    camera_data['detection_timestamp'] = time.time()

                    # Mark person as waiting if face recognized
                    if recognized_faces:
                        _, match = recognized_faces[0]
                        person_name = match['name']
                        face_id = match['id']

                        with self.person_tracking_lock:
                            # Check if we should start collection for this person
                            if self.should_start_collection(person_name, face_id):
                                # If same person or no one waiting, set/update waiting person
                                if self.waiting_person is None or self.waiting_person['name'] == person_name:
                                    if self.waiting_person is None:
                                        self.waiting_person = {
                                            'name': person_name,
                                            'face_id': face_id,
                                            'entered_zone_time': None
                                        }
                                        print(f"[WAITING] {person_name} recognized, waiting for them to enter zone...")
                                    # If same person, just refresh (keep existing state)
                                else:
                                    # Different person recognized while someone waiting - ignore for now
                                    print(f"[WAITING] {person_name} recognized but {self.waiting_person['name']} is already waiting")

                elif job.kind == 'PERSON':
                    # Person detection to track zone entry
                    t0 = time.time()
                    person_dets, _ = self.person_detector.detect_persons(job.frame)
                    infer_ms = (time.time() - t0) * 1000.0
                    self._stats['person_qwait_ms'].append(q_wait_ms)
                    self._stats['person_infer_ms'].append(infer_ms)

                    print(f"[PERSON] Detected {len(person_dets)} persons in {job.camera_id} camera")

                    # Check if waiting person is in zone
                    with self.person_tracking_lock:
                        if self.waiting_person is not None and person_dets:
                            best_person = max(person_dets, key=lambda p: p['confidence'])
                            person_name = self.waiting_person['name']

                            if self.is_person_in_zone(best_person['bbox']):
                                # Person entered zone
                                if self.waiting_person['entered_zone_time'] is None:
                                    self.waiting_person['entered_zone_time'] = time.time()
                                    print(f"[ZONE] {person_name} entered zone! Starting {self.countdown_duration}s countdown...")

                                # Check if countdown completed
                                elapsed = time.time() - self.waiting_person['entered_zone_time']
                                if elapsed >= self.countdown_duration:
                                    # Start collection!
                                    face_id = self.waiting_person['face_id']

                                    print(f"[COLLECTION] Countdown complete! Starting collection for {person_name}")

                                    with self.collection_lock:
                                        if self.active_collection is None:
                                            self.active_collection = {
                                                'person_name': person_name,
                                                'face_id': face_id,
                                                'start_time': time.time(),
                                                'embeddings': [],
                                                'visibility': [],
                                                'frames_collected': 0
                                            }

                                    # Switch mode to COLLECT
                                    with self.mode_lock:
                                        self.mode = Mode.COLLECT

                                    # Clear waiting person
                                    self.waiting_person = None
                                    print(f"[MODE] Switched to COLLECT mode for {person_name}")
                                else:
                                    print(f"[COUNTDOWN] {person_name} in zone: {elapsed:.1f}/{self.countdown_duration}s")
                            else:
                                # Person left zone, reset countdown
                                if self.waiting_person['entered_zone_time'] is not None:
                                    print(f"[ZONE] {person_name} left zone, resetting countdown")
                                    self.waiting_person['entered_zone_time'] = None

            else:  # COLLECT
                if job.kind != 'PERSON' or job.camera_id != 'camera':
                    continue

                # Detect persons in camera
                t0 = time.time()
                person_dets, _ = self.person_detector.detect_persons(job.frame)
                infer_ms = (time.time() - t0) * 1000.0
                self._stats['person_qwait_ms'].append(q_wait_ms)
                self._stats['person_infer_ms'].append(infer_ms)

                print(f"[PERSON] Detected {len(person_dets)} persons in camera")

                with self.collection_lock:
                    if not person_dets or self.active_collection is None:
                        continue

                    collection = self.active_collection
                    if collection['frames_collected'] >= self.target_embeddings_per_person:
                        continue

                # Pick best detection and embed
                best = max(person_dets, key=lambda p: p['confidence'])
                print(f"[PERSON] Best detection confidence: {best['confidence']:.3f} bbox: {best['bbox']}")

                crop = self.person_detector.get_person_crop(job.frame, best['bbox'], padding=10)
                if crop is None:
                    print(f"[CROP] Failed to extract person crop from {job.camera_id}")
                    continue

                print(f"[CROP] Extracted crop shape: {crop.shape} from {job.camera_id}")

                # Convert BGR to RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                t_emb_start = time.time()
                emb, visibility = self.person_embedder.extract_embedding(crop_rgb)
                emb_time = (time.time() - t_emb_start) * 1000.0

                if emb is None or visibility is None:
                    print(f"[EMBEDDING] Failed to extract embedding from {job.camera_id}")
                    continue

                print(f"[EMBEDDING] Extracted embedding shape: {emb.shape}, visibility shape: {visibility.shape}, took {emb_time:.1f}ms")

                # Collect embedding
                should_complete = False
                with self.collection_lock:
                    if self.active_collection is not None:
                        collection = self.active_collection
                        collection['embeddings'].append(emb.unsqueeze(0))
                        collection['visibility'].append(visibility.unsqueeze(0))
                        collection['frames_collected'] += 1
                        self.debug_counters['embedding_successes'] += 1

                        print(f"[COLLECT] Collected {collection['frames_collected']}/{self.target_embeddings_per_person} for {collection['person_name']} from {job.camera_id}")

                        # Update UI
                        camera_data = self.cameras[job.camera_id]
                        camera_data['persistent_detections'] = [{
                            'bbox': best['bbox'],
                            'match': {'name': collection['person_name']},
                            'confidence': best['confidence'],
                            'type': 'person_bpbreid'
                        }]
                        camera_data['detection_timestamp'] = time.time()

                        # Check if target reached
                        if collection['frames_collected'] >= self.target_embeddings_per_person:
                            should_complete = True

                # Complete collection outside of lock to avoid deadlock
                if should_complete:
                    self.complete_collection()

    def complete_collection(self):
        """Complete the active collection and save embeddings."""
        # Extract collection data with lock
        with self.collection_lock:
            if self.active_collection is None:
                return

            collection = self.active_collection
            person_name = collection['person_name']
            face_id = collection['face_id']
            embeddings_list = collection['embeddings']
            visibility_list = collection['visibility']

            print(f"[COLLECTION] Completing collection for {person_name}")

        # Save to .pt file (OVERWRITES if exists) - without holding lock
        if embeddings_list and visibility_list:
            all_embeddings = torch.cat(embeddings_list, dim=0)
            all_visibility = torch.cat(visibility_list, dim=0)

            # Save to memory
            self.live_gallery.add_person(
                pid=face_id,
                embeddings=all_embeddings,
                visibility=all_visibility
            )

            # Save to disk (OVERWRITE)
            self.save_person_embeddings_to_pt(person_name, embeddings_list, visibility_list, face_id)

        # Clear active collection with lock
        with self.collection_lock:
            self.active_collection = None

        # Switch back to FACES mode
        with self.mode_lock:
            self.mode = Mode.FACES
        print(f"[MODE] Switched back to FACES mode")

    def initialize_cameras(self):
        """Initialize camera pipelines."""
        for camera_id in ['camera']:
            config = self.cameras[camera_id]['config']
            print(f"Initializing camera: {config.get('name', config['url'])}")

            try:
                self.create_camera_pipeline(camera_id, config)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error initializing {camera_id} camera: {e}")

    def create_camera_pipeline(self, camera_id, config):
        """Create GStreamer pipeline for a camera."""
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
        """Handle new frame from GStreamer pipeline."""
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

                    # Store with thread safety
                    with self.cameras[camera_id]['frame_lock']:
                        self.cameras[camera_id]['latest_raw_frame'] = frame
                        self.cameras[camera_id]['frame_counter'] += 1

                    # Add to processing buffer
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
        """Handle GStreamer pipeline messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Pipeline error for {camera_id}: {err}")
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.cameras[camera_id]['pipeline']:
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    self.cameras[camera_id]['playing'] = True
                    print(f"Camera {camera_id} pipeline started")

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

                    with self.mode_lock:
                        mode = self.mode

                    should_detect = (frame_count % self.detection_interval == 0)

                    if mode == Mode.FACES:
                        # Camera sends FACE jobs for face recognition
                        if camera_id == 'camera' and should_detect:
                            try:
                                self.job_q.put_nowait(Job('FACE', camera_id, frame, time.time()))
                            except:
                                pass  # drop if queue full

                        # Camera sends PERSON jobs for zone tracking when someone is waiting
                        if camera_id == 'camera':
                            with self.person_tracking_lock:
                                has_waiting_person = self.waiting_person is not None

                            if has_waiting_person and should_detect:
                                try:
                                    self.job_q.put_nowait(Job('PERSON', camera_id, frame, time.time()))
                                except:
                                    pass  # drop if queue full

                    # Camera sends PERSON jobs in COLLECT mode
                    elif mode == Mode.COLLECT and camera_id == 'camera':
                        with self.collection_lock:
                            if self.active_collection is not None:
                                try:
                                    self.job_q.put_nowait(Job('PERSON', camera_id, frame, time.time()))
                                except:
                                    pass

                    # Always update processed frame for display
                    current_time = time.time()
                    if (current_time - camera_data['detection_timestamp']) < 2.0:
                        processed_frame = self.draw_detections(
                            frame, camera_data['persistent_detections'], camera_id
                        )
                    else:
                        processed_frame = self.draw_detections(frame, [], camera_id)

                    # Store processed frame
                    with camera_data['frame_lock']:
                        camera_data['latest_processed_frame'] = processed_frame

                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error in processing thread for {camera_id}: {e}")
                time.sleep(0.1)

    def draw_detections(self, frame, detections, camera_id):
        """Draw detection results on frame."""
        camera_data = self.cameras[camera_id]
        new_width, new_height, pad_left, pad_top, pad_right, pad_bottom = \
            calculate_display_dimensions(
                frame.shape[1], frame.shape[0],
                self.camera_width, self.camera_height
            )

        # Resize frame maintaining aspect ratio
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create padded frame
        display_frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        display_frame[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = resized_frame

        # Calculate scaling factors
        scale_x = new_width / frame.shape[1]
        scale_y = new_height / frame.shape[0]

        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            match = detection['match']
            detection_type = detection.get('type', 'face')

            # Scale and offset coordinates
            x1 = int(bbox[0] * scale_x) + pad_left
            y1 = int(bbox[1] * scale_y) + pad_top
            x2 = int(bbox[2] * scale_x) + pad_left
            y2 = int(bbox[3] * scale_y) + pad_top

            # Choose color and label
            if match:
                person_name = match['name']

                if detection_type == 'person_bpbreid':
                    embedding_status = f"COLLECTING"
                    color = (255, 165, 0)  # Orange
                    thickness = 4
                    label = f"{person_name} {embedding_status}"
                else:
                    embedding_status = f"RECOGNIZED"
                    color = (0, 255, 0)  # Green
                    thickness = 3
                    label = f"{person_name} {embedding_status}"
            else:
                color = (0, 0, 255)
                label = "Unknown"
                thickness = 3

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw designated zone polygon (scaled to display coordinates)
        zone_points = []
        for point in self.designated_zone:
            x_scaled = int(point[0] * scale_x) + pad_left
            y_scaled = int(point[1] * scale_y) + pad_top
            zone_points.append([x_scaled, y_scaled])
        zone_points = np.array(zone_points, dtype=np.int32)

        # Draw zone with different colors based on state
        with self.person_tracking_lock:
            if self.waiting_person is not None:
                if self.waiting_person['entered_zone_time'] is not None:
                    # Person in zone - countdown active (green)
                    zone_color = (0, 255, 0)
                    zone_thickness = 3
                else:
                    # Person recognized but not in zone yet (yellow)
                    zone_color = (0, 255, 255)
                    zone_thickness = 2
            else:
                # No waiting person (magenta)
                zone_color = (255, 0, 255)
                zone_thickness = 2

        cv2.polylines(display_frame, [zone_points], isClosed=True, color=zone_color, thickness=zone_thickness)

        # Add countdown text if person is in zone
        with self.person_tracking_lock:
            if self.waiting_person is not None and self.waiting_person['entered_zone_time'] is not None:
                elapsed = time.time() - self.waiting_person['entered_zone_time']
                remaining = max(0, self.countdown_duration - elapsed)
                person_name = self.waiting_person['name']

                countdown_text = f"{person_name}: {remaining:.1f}s"
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]

                # Calculate center of zone for text placement
                zone_center_x = int(np.mean(zone_points[:, 0]))
                zone_center_y = int(np.mean(zone_points[:, 1]))

                text_x = zone_center_x - text_size[0] // 2
                text_y = zone_center_y + text_size[1] // 2

                # Draw text background
                cv2.rectangle(display_frame,
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            (0, 0, 0), -1)

                # Draw countdown text
                cv2.putText(display_frame, countdown_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Add camera info
        camera_name = camera_data['config'].get('name', 'Camera')

        cv2.putText(display_frame, f"{camera_name}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Add mode indicator
        with self.mode_lock:
            mode_text = f"Mode: {self.mode.name}"
            mode_color = (255, 255, 0) if self.mode == Mode.COLLECT else (128, 128, 255)
        cv2.putText(display_frame, mode_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        # Add active collection info
        with self.collection_lock:
            if self.active_collection:
                active_text = f"Collecting: {self.active_collection['person_name']} ({self.active_collection['frames_collected']}/{self.target_embeddings_per_person})"
                cv2.putText(display_frame, active_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Add waiting person status
        with self.person_tracking_lock:
            if self.waiting_person is not None:
                person_name = self.waiting_person['name']
                if self.waiting_person['entered_zone_time'] is None:
                    wait_text = f"WAITING: {person_name} - Enter Zone"
                    wait_color = (0, 255, 255)
                else:
                    wait_text = f"COUNTDOWN: {person_name} in zone"
                    wait_color = (0, 255, 0)
                cv2.putText(display_frame, wait_text, (10, 160),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, wait_color, 2)

        # Add FPS
        fps_text = f"FPS: {self.current_fps[camera_id]}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display_frame

    def calculate_fps(self, camera_id):
        """Calculate and update FPS for a camera."""
        self.fps_counters[camera_id] += 1
        if time.time() - self.fps_start_times[camera_id] >= 1.0:
            self.current_fps[camera_id] = self.fps_counters[camera_id]
            self.fps_counters[camera_id] = 0
            self.fps_start_times[camera_id] = time.time()

    def update_stable_display(self):
        """Update stable display buffer."""
        with self.display_lock:
            updated = False

            for camera_id in ['camera']:
                camera_data = self.cameras[camera_id]

                with camera_data['frame_lock']:
                    if camera_data['latest_processed_frame'] is not None:
                        self.stable_display_buffer[camera_id] = camera_data['latest_processed_frame'].copy()
                        updated = True
                        self.calculate_fps(camera_id)

            if updated:
                self.stable_display_buffer['timestamp'] = time.time()

    def create_stable_dual_display(self):
        """Create single camera display."""
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        with self.display_lock:
            # Single camera display
            if self.stable_display_buffer['camera'] is not None:
                canvas[0:self.camera_height, 0:self.camera_width] = self.stable_display_buffer['camera']
            else:
                placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting Camera...", (50, self.camera_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                canvas[0:self.camera_height, 0:self.camera_width] = placeholder

        return canvas

    def start_collection(self):
        """Start the collection system."""
        # Start pipelines
        for camera_id in ['camera']:
            try:
                pipeline = self.cameras[camera_id]['pipeline']
                ret = pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    print(f"Failed to start camera")
                time.sleep(1.0)
            except Exception as e:
                print(f"Error starting camera: {e}")

        time.sleep(3)
        self.running = True

        # Start GPU worker
        self.gpu_worker_thread = threading.Thread(target=self.gpu_worker_loop, args=(), daemon=True)
        self.gpu_worker_thread.start()

        # Start processing threads
        for camera_id in ['camera']:
            thread = threading.Thread(target=self.processing_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.cameras[camera_id]['thread'] = thread
            time.sleep(0.2)

        try:
            while self.running:
                # Update display
                self.update_stable_display()

                # Show display
                display = self.create_stable_dual_display()
                if not os.getenv('HEADLESS'):
                    cv2.imshow('Single Camera Person Embedding Collector', display)

                # Handle controls
                key = cv2.waitKey(33) & 0xFF if not os.getenv('HEADLESS') else -1
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_system()

    def stop_system(self):
        """Stop the collection system."""
        self.running = False

        # Complete any active collection
        with self.collection_lock:
            if self.active_collection:
                self.complete_collection()

        # Stop camera pipelines
        for camera_id in ['camera']:
            try:
                pipeline = self.cameras[camera_id]['pipeline']
                if pipeline:
                    pipeline.set_state(Gst.State.NULL)

                thread = self.cameras[camera_id].get('thread')
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
            except Exception as e:
                print(f"Error stopping camera: {e}")

        cv2.destroyAllWindows()
