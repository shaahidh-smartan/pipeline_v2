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

class Mode(Enum):
    FACES = 0
    COLLECT = 1

@dataclass
class Job:
    kind: str              # 'FACE' | 'PERSON'
    camera_id: str
    frame: np.ndarray
    ts: float


class PersonEmbeddingCollector:
    """
    Person Embedding Collector Service for multi-camera person re-identification.

    Collects BPBreID person embeddings from 4 cameras using face recognition triggers.
    Uses BPBreID (Body Part-Based Re-Identification) for high-quality person embeddings.
    """
    
    def __init__(self, stream_configs, similarity_threshold=0.6, target_embeddings_per_person=25):
        """
        Initialize Person Embedding Collector for multi-camera person re-identification.

        Args:
            stream_configs (list): List of 4 camera configuration dictionaries
            similarity_threshold (float): Face matching threshold (default: 0.6)
            target_embeddings_per_person (int): Target embeddings per person per camera (default: 25)

        Returns:
            None
        """
        if len(stream_configs) != 2:
            raise ValueError("This system is designed for exactly 2 cameras (center and right)")
            
        self.stream_configs = stream_configs
        self.similarity_threshold = similarity_threshold
        self.target_embeddings_per_person = target_embeddings_per_person
        
        # Initialize database first (no GPU dependencies)
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
        
        # Display configuration (side-by-side for 2 cameras)
        self.camera_width = 640
        self.camera_height = 480
        self.display_width = 2 * self.camera_width  # 2 cameras side by side
        self.display_height = self.camera_height

        # Processing configuration - Frame-based collection
        self.detection_interval = 3  # Process every 3rd frame for detection
        
        # Initialize cameras (only center and right)
        self.cameras = {
            'center': create_camera_structure(stream_configs[0]),
            'right': create_camera_structure(stream_configs[1])
        }
        
        # Control flags
        self.running = False
        self.collecting_embeddings = True  # START WITH COLLECTION ON
        
        # Frame synchronization
        self.display_lock = threading.Lock()
        self.stable_display_buffer = {
            'center': None,
            'right': None,
            'timestamp': time.time()
        }
        
        # Frame-based embedding collection tracking
        self.frame_collection_lock = threading.Lock()
        self.active_collections = {}  # person_name: collection_data
        
        # Inference mode (mutually exclusive)
        self.mode = Mode.FACES
        self.mode_lock = threading.Lock()

        # Bounded GPU job queue
        self.job_q = Queue(maxsize=64)

        # For worker metrics (rolling averages)
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
            'face_person_matches': 0,
            'embedding_attempts': 0,
            'embedding_successes': 0
        }
        
        # Performance tracking
        self.fps_counters = {'center': 0, 'right': 0}
        self.fps_start_times = {'center': time.time(), 'right': time.time()}
        self.current_fps = {'center': 0, 'right': 0}
        
        # Initialize pipelines
        self.initialize_cameras()
        
        print("PersonEmbeddingCollector initialized - GPU models will load in worker thread")

    def gpu_worker_loop(self):
        """
        GPU worker thread that processes face detection and person embedding jobs.

        Args:
            None

        Returns:
            None
        """
        # --- Construct models once, warmup once ---
        print("[GPU] Initializing FacePipeline/PersonDetector/PersonEmbedder...")
        self.face_pipeline = FacePipeline()
        self.person_detector = PersonDetector()
        self.person_embedder = PersonEmbedder()

        # Warmup
        try:
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_crop  = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            _ = self.face_pipeline.process_frame(dummy_frame, thresh=0.5, input_size=(640, 640))
            _ = self.person_detector.detect_persons(dummy_frame)
            _, _ = self.person_embedder.extract_embedding(dummy_crop)  # Now returns (embedding, visibility)
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
                # Only accept FACE jobs from right camera
                if job.kind != 'FACE' or job.camera_id != 'right':
                    continue

                t0 = time.time()
                face_results = self.face_pipeline.process_frame(job.frame, thresh=0.5, input_size=(640, 640))
                infer_ms = (time.time() - t0) * 1000.0
                self._stats['face_qwait_ms'].append(q_wait_ms)
                self._stats['face_infer_ms'].append(infer_ms)

                # Recognize & start collection
                recognized_faces = []
                self.debug_counters['face_detections'] += len(face_results)
                print(f"[FACE] Detected {len(face_results)} faces in {job.camera_id} camera")

                for face_data in face_results:
                    if face_data.get('embedding') is None:
                        continue
                    match, sim = self.similarity_search.find_face_match_euclidean(
                        face_data['embedding'], threshold=self.similarity_threshold
                    )
                    if match:
                        self.debug_counters['recognized_faces'] += 1
                        recognized_faces.append((face_data, match))
                        print(f"[FACE] Recognized: {match['name']} (similarity: {sim:.3f})")

                if recognized_faces:
                    # Start a collection for the first recognized person
                    _, match = recognized_faces[0]
                    person_name = match['name']
                    face_id = match['id']  # Get face_id from the match result
                    print(f"[COLLECTION] Checking if should start collection for {person_name} (face_id={face_id})")
                    if self.should_start_frame_collection(person_name):
                        print(f"[COLLECTION] Starting collection for {person_name}")
                        self.start_frame_collection(person_name, face_id, job.camera_id)
                        # Switch mode to COLLECT
                        with self.mode_lock:
                            self.mode = Mode.COLLECT
                        print(f"[MODE] Switched to COLLECT mode for {person_name}")
                    else:
                        print(f"[COLLECTION] Skipping collection for {person_name} (already complete or in progress)")

            else:  # COLLECT
                if job.kind != 'PERSON':
                    continue

                # Detect persons in this frame
                t0 = time.time()
                person_dets, _ = self.person_detector.detect_persons(job.frame)
                infer_ms = (time.time() - t0) * 1000.0
                self._stats['person_qwait_ms'].append(q_wait_ms)
                self._stats['person_infer_ms'].append(infer_ms)

                print(f"[PERSON] Detected {len(person_dets)} persons in {job.camera_id} camera")

                if not person_dets or not self.active_collections:
                    if not person_dets:
                        print(f"[PERSON] No person detections in {job.camera_id}")
                    continue

                # For each active collection, try to collect from this camera
                for person_name in list(self.active_collections.keys()):
                    with self.frame_collection_lock:
                        if person_name not in self.active_collections:
                            continue
                        cam_data = self.active_collections[person_name]['cameras'][job.camera_id]
                        if cam_data['frames_collected'] >= self.target_embeddings_per_person:
                            continue

                    # Pick best detection and embed inline
                    best = max(person_dets, key=lambda p: p['confidence'])
                    print(f"[PERSON] Best detection confidence: {best['confidence']:.3f} bbox: {best['bbox']}")

                    crop = self.person_detector.get_person_crop(job.frame, best['bbox'], padding=10)
                    if crop is None:
                        print(f"[CROP] Failed to extract person crop from {job.camera_id}")
                        continue

                    print(f"[CROP] Extracted crop shape: {crop.shape} from {job.camera_id}")

                    t_emb_start = time.time()
                    emb, visibility = self.person_embedder.extract_embedding(crop)
                    emb_time = (time.time() - t_emb_start) * 1000.0

                    if emb is None or visibility is None:
                        print(f"[EMBEDDING] Failed to extract embedding for {person_name} from {job.camera_id}")
                        continue

                    print(f"[EMBEDDING] Extracted embedding shape: {emb.shape}, visibility shape: {visibility.shape}, took {emb_time:.1f}ms for {person_name} from {job.camera_id}")

                    # Get face_id for this person from the active collection data
                    with self.frame_collection_lock:
                        if person_name not in self.active_collections:
                            continue
                        face_id = self.active_collections[person_name]['face_id']

                    # Save to DB using BPBreID embedding table
                    print(f"[DB] Attempting to save BPBreID embedding for {person_name} (face_id={face_id}) from {job.camera_id}")
                    if self.db_manager.insert_bpbreid_embedding(face_id, emb, visibility, job.camera_id):
                        print(f"[DB] Successfully saved embedding for {person_name} from {job.camera_id}")
                        with self.frame_collection_lock:
                            if person_name in self.active_collections:  # Double-check still active
                                cdata = self.active_collections[person_name]
                                cdata['cameras'][job.camera_id]['frames_collected'] += 1
                                cdata['cameras'][job.camera_id]['last_person_box'] = best['bbox']
                                cdata['total_frames_collected'] = sum(
                                    v['frames_collected'] for v in cdata['cameras'].values()
                                )
                                self.debug_counters['embedding_successes'] += 1

                                # If all cameras hit target, complete
                                done = all(v['frames_collected'] >= self.target_embeddings_per_person
                                           for v in cdata['cameras'].values())

                                if done:
                                    self.complete_frame_collection(person_name)
                        
                        # Update UI to show person bounding box being used for BPBreID embedding
                        camera_data = self.cameras[job.camera_id]
                        camera_data['persistent_detections'] = [{
                            'bbox': best['bbox'],
                            'match': {'name': person_name},
                            'confidence': best['confidence'],
                            'type': 'person_bpbreid'  # Mark as person detection for BPBreID
                        }]
                        camera_data['detection_timestamp'] = time.time()
                        
                        break  # Only collect once per frame

                # If no active collections left → switch back to FACES
                with self.frame_collection_lock:
                    if not self.active_collections:
                        with self.mode_lock:
                            self.mode = Mode.FACES

    def initialize_cameras(self):
        """
        Initialize all camera pipelines sequentially in GStreamer.

        Args:
            None

        Returns:
            None
        """
        for camera_id in ['center', 'right']:
            config = self.cameras[camera_id]['config']
            print(f"Initializing {camera_id} camera: {config.get('name', config['url'])}")
            
            try:
                self.create_camera_pipeline(camera_id, config)
                # GStreamer documentation: Allow pipeline to stabilize before next
                time.sleep(0.5)
            except Exception as e:
                print(f"Error initializing {camera_id} camera: {e}")
    
    def create_camera_pipeline(self, camera_id, config):
        """
        Create GStreamer pipeline for a specific camera.

        Args:
            camera_id (str): Camera identifier ('left', 'right', 'center', 'back')
            config (dict): Camera configuration dictionary

        Returns:
            None
        """
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
        """
        Handle new frame from GStreamer pipeline.

        Args:
            appsink: GStreamer appsink element
            camera_id (str): Camera identifier

        Returns:
            Gst.FlowReturn: GStreamer flow return status
        """
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
        """
        Handle GStreamer pipeline messages.

        Args:
            bus: GStreamer bus
            message: GStreamer message
            camera_id (str): Camera identifier

        Returns:
            bool: True to continue handling messages
        """
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
    
    def should_start_frame_collection(self, person_name):
        """
        Check if frame collection should start for a person.

        Args:
            person_name (str): Name of the person

        Returns:
            bool: True if collection should start, False otherwise
        """
        # Get current count from database
        db_count = self.db_manager.get_person_embedding_count(person_name)

        # Check if person already has complete collection
        if db_count >= self.target_embeddings_per_person * 2:  # 25 per camera × 2 cameras = 50
            return False
        
        # Check if already collecting for this person on ANY camera
        if person_name in self.active_collections:
            return False
        
        return True
    
    def start_frame_collection(self, person_name, face_id, trigger_camera_id):
        """
        Initialize per-camera counters for a new collection.

        Args:
            person_name (str): Name of the person
            face_id (int): Face ID from face_embeddings table
            trigger_camera_id (str): Camera that triggered the collection

        Returns:
            bool: True if collection started, False otherwise
        """
        with self.frame_collection_lock:
            if person_name not in self.active_collections:
                self.active_collections[person_name] = {
                    'face_id': face_id,  # Store face_id for BPBreID embedding insertion
                    'trigger_camera': trigger_camera_id,
                    'start_time': time.time(),
                    'cameras': {
                        'center': {'frames_collected': 0, 'embeddings': [], 'last_person_box': None},
                        'right': {'frames_collected': 0, 'embeddings': [], 'last_person_box': None}
                    },
                    'total_frames_collected': 0
                }
                print(f"Started frame collection for {person_name} (face_id={face_id}) on ALL cameras (triggered by {trigger_camera_id})")
                return True
        return False

    def complete_frame_collection(self, person_name):
        """
        Finalize a collection and update database counters.

        Args:
            person_name (str): Name of the person

        Returns:
            None
        """
        if person_name not in self.active_collections:
            return
        
        collection_data = self.active_collections[person_name]
        
        # Get final count from database
        final_count = self.db_manager.get_person_embedding_count(person_name)
        
        # Remove from active collections
        del self.active_collections[person_name]
        
        # Update debug counters
        total_frames = collection_data['total_frames_collected']
        self.debug_counters['embedding_successes'] += total_frames
        
        camera_summary = ', '.join([
            f"{cam}: {cam_data['frames_collected']}" 
            for cam, cam_data in collection_data['cameras'].items() 
            if cam_data['frames_collected'] > 0
        ])
        
        print(f"Completed frame collection for {person_name}: "
              f"{final_count} embeddings saved to database")
    
    def cleanup_stale_collections(self):
        """
        Clean up collections that have been running too long.

        Args:
            None

        Returns:
            None
        """
        current_time = time.time()
        stale_collections = []
        
        with self.frame_collection_lock:
            for person_name, collection_data in self.active_collections.items():
                # If collection is running for more than 30 seconds, consider it stale
                if current_time - collection_data['start_time'] > 30:
                    stale_collections.append(person_name)
            
            for person_name in stale_collections:
                # Check if we have any embeddings in the database for this person
                db_count = self.db_manager.get_person_embedding_count(person_name)
                
                if db_count > 0:
                    self.complete_frame_collection(person_name)
                else:
                    del self.active_collections[person_name]
    
    def processing_thread(self, camera_id):
        """
        Camera processing thread that captures frames and enqueues GPU work.

        Args:
            camera_id (str): Camera identifier

        Returns:
            None
        """
        while self.running:
            try:
                camera_data = self.cameras[camera_id]
                frame_buffer = camera_data['frame_buffer']
                
                if not frame_buffer.empty():
                    frame = frame_buffer.get()
                    
                    with camera_data['frame_lock']:
                        frame_count = camera_data['frame_counter']
                    
                    # In FACES mode: only RIGHT camera sends FACE jobs at interval
                    with self.mode_lock:
                        mode = self.mode

                    should_detect = (frame_count % self.detection_interval == 0)

                    if mode == Mode.FACES:
                        if camera_id == 'right' and should_detect:
                            try:
                                self.job_q.put_nowait(Job('FACE', camera_id, frame, time.time()))
                            except:
                                pass  # drop if queue full
                    else:  # COLLECT
                        if self.collecting_embeddings and self.active_collections:
                            try:
                                # All cameras propose PERSON frames during COLLECT
                                self.job_q.put_nowait(Job('PERSON', camera_id, frame, time.time()))
                            except:
                                pass
                    
                    # Always update the processed frame for display
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
        """
        Draw detection results on frame with bounding boxes and labels.

        Args:
            frame (np.ndarray): Input frame
            detections (list): List of detection dictionaries
            camera_id (str): Camera identifier

        Returns:
            np.ndarray: Frame with drawn detections
        """
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
        
        # Draw detections (both face and person)
        for detection in detections:
            bbox = detection['bbox']
            match = detection['match']
            detection_type = detection.get('type', 'face')  # Default to face detection
            
            # Scale and offset coordinates
            x1 = int(bbox[0] * scale_x) + pad_left
            y1 = int(bbox[1] * scale_y) + pad_top
            x2 = int(bbox[2] * scale_x) + pad_left
            y2 = int(bbox[3] * scale_y) + pad_top
            
            # Choose color and label based on detection type
            if match:
                person_name = match['name']
                
                if detection_type == 'person_bpbreid':
                    # Person bounding box for BPBreID embedding collection
                    embedding_status = f"BPBREID EMBEDDING"
                    color = (255, 165, 0)  # Orange for BPBreID person detection
                    thickness = 4  # Thicker for visibility
                    label = f"{person_name} {embedding_status}"
                else:
                    # Face detection box
                    if person_name in self.active_collections:
                        embedding_status = f"COLLECTING [DB]"
                        color = (0, 255, 255)  # Cyan when actively collecting
                    else:
                        embedding_status = f"RECOGNIZED [DB]"
                        color = (0, 255, 0)  # Green for recognized
                    thickness = 3
                    label = f"{person_name} {embedding_status}"
            else:
                color = (0, 0, 255)
                label = "Unknown"
                thickness = 3
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add camera info
        camera_name = camera_data['config'].get('name', f'Camera {camera_id.title()}')
        if camera_id == 'right':
            camera_role = "PRIMARY"
        else:
            camera_role = "SECONDARY"
        
        cv2.putText(display_frame, f"{camera_name} ({camera_role})", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add GPU-safe status
        status_color = (0, 255, 0) if self.collecting_embeddings else (0, 0, 255)
        status_text = f"Collection: {'ON' if self.collecting_embeddings else 'OFF'} [GPU-SAFE]"
        cv2.putText(display_frame, status_text, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Add active collections count
        active_count = len(self.active_collections)
        if active_count > 0:
            active_text = f"Active Collections: {active_count}"
            cv2.putText(display_frame, active_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add mode indicator
        with self.mode_lock:
            mode_text = f"Mode: {self.mode.name}"
            mode_color = (255, 255, 0) if self.mode == Mode.COLLECT else (128, 128, 255)
        cv2.putText(display_frame, mode_text, (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Add FPS
        fps_text = f"FPS: {self.current_fps[camera_id]}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return display_frame
    
    def calculate_fps(self, camera_id):
        """
        Calculate and update FPS for a camera.

        Args:
            camera_id (str): Camera identifier

        Returns:
            None
        """
        self.fps_counters[camera_id] += 1
        if time.time() - self.fps_start_times[camera_id] >= 1.0:
            self.current_fps[camera_id] = self.fps_counters[camera_id]
            self.fps_counters[camera_id] = 0
            self.fps_start_times[camera_id] = time.time()
    
    def update_stable_display(self):
        """
        Update stable display buffer with latest processed frames.

        Args:
            None

        Returns:
            None
        """
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
        """
        Create a side-by-side display for 2 cameras (center and right).

        Args:
            None

        Returns:
            np.ndarray: Canvas with 2 cameras side by side
        """
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        with self.display_lock:
            # Left side - Center camera
            if self.stable_display_buffer['center'] is not None:
                canvas[0:self.camera_height, 0:self.camera_width] = self.stable_display_buffer['center']
            else:
                placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting Center Camera...", (50, self.camera_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                canvas[0:self.camera_height, 0:self.camera_width] = placeholder

            # Right side - Right camera
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
        """
        Draw global status information on canvas.

        Args:
            canvas (np.ndarray): Canvas to draw on

        Returns:
            None
        """
        y_pos = 200
        line_height = 35
        
        # Draw person embedding collection summary
        cv2.putText(canvas, "BPBreID Person Embedding Collection (DATABASE):",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height
        
        # Create thread-safe copy of active_collections
        active_collections_copy = {}
        with self.frame_collection_lock:
            active_collections_copy = dict(self.active_collections)
        
        # Show active collections first
        if active_collections_copy:
            cv2.putText(canvas, "ACTIVE COLLECTIONS:", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += line_height - 10
            
            for person_name, collection_data in active_collections_copy.items():
                center_frames = collection_data['cameras']['center']['frames_collected']
                right_frames = collection_data['cameras']['right']['frames_collected']
                elapsed = time.time() - collection_data['start_time']

                # Get current DB count
                db_count = self.db_manager.get_person_embedding_count(person_name)

                # Target is 25 per camera * 2 cameras = 50
                active_text = f"  {person_name}: {db_count}/50 in DB [C:{center_frames}, R:{right_frames}] ({elapsed:.1f}s)"
                cv2.putText(canvas, active_text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_pos += line_height - 10
        
        # Add debug statistics
        y_pos += 10
        stats_text = f"Debug: Faces:{self.debug_counters['face_detections']} | Recognized:{self.debug_counters['recognized_faces']} | Persons:{self.debug_counters['person_detections']} | Matches:{self.debug_counters['face_person_matches']}"
        cv2.putText(canvas, stats_text, (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
   
    def start_embedding_collection(self):
        """
        Start the person embedding collection system.

        Args:
            None

        Returns:
            None
        """
        # Start pipelines sequentially (GStreamer best practice)
        for camera_id in ['center', 'right']:
            try:
                print(f"[DEBUG] About to start {camera_id} pipeline...")
                pipeline = self.cameras[camera_id]['pipeline']
                print(f"[DEBUG] Setting {camera_id} to PLAYING state...")
                ret = pipeline.set_state(Gst.State.PLAYING)
                print(f"[DEBUG] {camera_id} set_state returned: {ret}")
                if ret == Gst.StateChangeReturn.FAILURE:
                    print(f"Failed to start {camera_id} camera")
                # GStreamer documentation: serialize pipeline startup
                print(f"[DEBUG] Waiting after {camera_id} start...")
                time.sleep(1.0)
            except Exception as e:
                print(f"Error starting {camera_id}: {e}")
                import traceback
                traceback.print_exc()
        
        time.sleep(3)  # Wait for initialization
        self.running = True
        
        # Start GPU worker first
        self.gpu_worker_thread = threading.Thread(target=self.gpu_worker_loop, args=(), daemon=True)
        self.gpu_worker_thread.start()
        
        # Start processing threads
        for camera_id in ['center', 'right']:
            thread = threading.Thread(target=self.processing_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.cameras[camera_id]['thread'] = thread
            # Small delay to stagger thread startup
            time.sleep(0.2)
        
        try:
            while self.running:
                # Update display buffer
                self.update_stable_display()
                
                # Create and show display
                display = self.create_stable_dual_display()
                if not os.getenv('HEADLESS'):
                    cv2.imshow('Person Embedding Collection', display)
                
                # Handle controls
                key = cv2.waitKey(33) & 0xFF if not os.getenv('HEADLESS') else -1
                if key == ord('q'):
                    break                
                # Cleanup stale collections periodically
                if time.time() % 10 < 0.1:
                    self.cleanup_stale_collections()
        
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_system()
    
    def stop_system(self):
        """
        Stop the person embedding collection system.

        Args:
            None

        Returns:
            None
        """
        self.running = False
        
        # Complete any active collections
        with self.frame_collection_lock:
            if self.active_collections:
                for person_name in list(self.active_collections.keys()):
                    self.complete_frame_collection(person_name)
        
        # Stop camera pipelines
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