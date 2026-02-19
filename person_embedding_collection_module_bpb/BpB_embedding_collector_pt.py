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
import random
import pickle
import hashlib
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path to access utils and core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redis configuration for embedding communication
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_EMBEDDING_STREAM = "person_embeddings"

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


def user_id_to_pid(user_id):
    """Convert a firebase user ID string to a stable numeric PID for tensors."""
    return int(hashlib.sha256(user_id.encode()).hexdigest()[:15], 16)


def apply_brightness_augmentation(crop, brightness_factor=1.5):
    """
    Apply high brightness augmentation to a crop.

    Args:
        crop (np.ndarray): RGB image crop
        brightness_factor (float): Brightness multiplier (1.5 = 50% brighter)

    Returns:
        np.ndarray: Brightened crop
    """
    # Convert to float for manipulation
    augmented = crop.astype(np.float32) * brightness_factor
    # Clip to valid range
    augmented = np.clip(augmented, 0, 255).astype(np.uint8)
    return augmented


def apply_rotation_augmentation(crop, max_angle=5):
    """
    Apply slight rotation augmentation to a crop.

    Args:
        crop (np.ndarray): RGB image crop
        max_angle (int): Maximum rotation angle in degrees (both directions)

    Returns:
        np.ndarray: Rotated crop
    """
    angle = random.uniform(-max_angle, max_angle)
    h, w = crop.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    augmented = cv2.warpAffine(crop, rotation_matrix, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    return augmented


def apply_noise_augmentation(crop, noise_level=15):
    """
    Apply Gaussian noise augmentation to a crop.

    Args:
        crop (np.ndarray): RGB image crop
        noise_level (int): Standard deviation of Gaussian noise

    Returns:
        np.ndarray: Noisy crop
    """
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, crop.shape).astype(np.float32)

    # Add noise to image
    augmented = crop.astype(np.float32) + noise

    # Clip to valid range
    augmented = np.clip(augmented, 0, 255).astype(np.uint8)
    return augmented


class PersonEmbeddingCollector:
    """
    Person Embedding Collector Service for multi-camera person re-identification.

    """

    def __init__(self, stream_configs, similarity_threshold=0.6, target_embeddings_per_person=5, output_dir=None):
        """
        Initialize Person Embedding Collector for multi-camera person re-identification.

        Args:
            stream_configs (list): List of 4 camera configuration dictionaries
            similarity_threshold (float): Face matching threshold (default: 0.6)
            target_embeddings_per_person (int): Target embeddings per person per camera (default: 5)
            output_dir (str): Directory to save .pt files (default: script_directory/embeddings_output)

        Returns:
            None
        """
        if len(stream_configs) != 2:
            raise ValueError("This system is designed for exactly 2 cameras (center and right)")

        self.stream_configs = stream_configs
        self.similarity_threshold = similarity_threshold
        self.target_embeddings_per_person = target_embeddings_per_person
        self.live_gallery = LiveGallery(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize Redis client for embedding communication
        self.redis_client = None
        self._init_redis_connection()

        # Set output directory - use shared location in repo root if not provided
        if output_dir is None:
            # Use repo root for shared access between modules
            repo_root = Path(__file__).parent.parent.absolute()
            self.output_dir = repo_root / 'embeddings_output'
        else:
            self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database for face recognition only
        self.db_manager = DatabaseManager()

        # Initialize CPU-side utilities only
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

        # Designated zone for person positioning (normalized coordinates for 640x480)
        self.designated_zone =  np.array([[685, 676], [741, 538], [1011, 554], [1022, 695]])

        # Person tracking for zone-based collection
        self.person_tracking_lock = threading.Lock()
        self.waiting_person = None  # {'name': str, 'face_id': int, 'entered_zone_time': float, 'camera_id': str}
        self.countdown_duration = 2.0  # seconds to wait before starting collection

        # Performance tracking
        self.fps_counters = {'center': 0, 'right': 0}
        self.fps_start_times = {'center': time.time(), 'right': time.time()}
        self.current_fps = {'center': 0, 'right': 0}

        # Initialize pipelines
        self.initialize_cameras()


    def _init_redis_connection(self):
        """Initialize Redis connection for embedding communication."""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=False,  # Keep binary mode for pickle
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis_client.ping()
        except ImportError:
            self.redis_client = None
        except Exception as e:
            self.redis_client = None

    def publish_embeddings_to_redis(self, person_name, embeddings, visibility, pids, face_id):
        """
        Publish embeddings to Redis stream for main pipeline consumption.

        Args:
            person_name (str): Name of the person
            embeddings (torch.Tensor): Embedding tensor [N, 6, 512]
            visibility (torch.Tensor): Visibility tensor [N, 6]
            pids (torch.Tensor): Person IDs tensor [N]
            face_id (str): Firebase user ID from firebase_users table

        Returns:
            bool: True if published successfully, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            # Move tensors to CPU for serialization
            embeddings_cpu = embeddings.cpu() if embeddings.is_cuda else embeddings
            visibility_cpu = visibility.cpu() if visibility.is_cuda else visibility
            pids_cpu = pids.cpu() if pids.is_cuda else pids

            # Create payload
            payload = {
                'person_name': person_name,
                'face_id': face_id,
                'embeddings': embeddings_cpu,
                'visibility': visibility_cpu,
                'pids': pids_cpu,
                'timestamp': time.time()
            }

            # Serialize with pickle
            serialized_data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

            # Publish to Redis stream
            self.redis_client.xadd(
                REDIS_EMBEDDING_STREAM,
                {'data': serialized_data},
                maxlen=100,  # Keep last 100 entries
                approximate=True
            )


            return True

        except Exception as e:
            return False

    def update_embedding_status(self, person_id, person_name, pt_path, status='READY'):
        """
        Update the embedding_status table in the database.

        Args:
            person_id (str): Firebase user ID from firebase_users table
            person_name (str): Name of the person
            pt_path (str): Path to the .pt file directory
            status (str): Status of the embedding ('PENDING', 'READY', 'CONSUMED')

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            current_timestamp = int(time.time() * 1000)  # milliseconds

            conn = self.db_manager.get_connection()
            if not conn:
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

            return True

        except Exception as e:
            return False

    def save_person_embeddings_to_pt(self, person_name, embeddings_list, visibility_list, face_id):
        """
        Save all collected embeddings for a person to a single .pt file.

        Args:
            person_name (str): Name of the person
            embeddings_list (list): List of embedding tensors
            visibility_list (list): List of visibility tensors
            face_id (str): Firebase user ID from firebase_users table (converted to numeric PID)

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            if not embeddings_list or not visibility_list:
                return False

            # Create person-specific directory using firebase user ID
            person_dir = self.output_dir / face_id
            person_dir.mkdir(parents=True, exist_ok=True)

            # Concatenate all embeddings and visibility scores
            all_embeddings = torch.cat(embeddings_list, dim=0)
            all_visibility = torch.cat(visibility_list, dim=0)

            # Move to CPU if on GPU to ensure portability
            all_embeddings = all_embeddings.cpu() if all_embeddings.is_cuda else all_embeddings
            all_visibility = all_visibility.cpu() if all_visibility.is_cuda else all_visibility

            # Create PIDs tensor (all samples belong to same person, identified by firebase user ID)
            num_samples = all_embeddings.shape[0]
            numeric_pid = user_id_to_pid(face_id)
            pids = torch.tensor([numeric_pid] * num_samples, dtype=torch.long)

            # Save embeddings, visibility, and PIDs as separate files (like build_gallery_from_videos.py)
            torch.save(all_embeddings, person_dir / 'embeddings.pt')
            torch.save(all_visibility, person_dir / 'visibility.pt')
            torch.save(pids, person_dir / 'pids.pt')


            # Publish embeddings to Redis for immediate consumption by main pipeline
            self.publish_embeddings_to_redis(
                person_name=person_name,
                embeddings=all_embeddings,
                visibility=all_visibility,
                pids=pids,
                face_id=face_id
            )

            return True

        except Exception as e:
            return False

    def check_person_embeddings_exist(self, face_id):
        """
        Check if embeddings already exist for a person.

        Args:
            face_id (str): Firebase user ID

        Returns:
            bool: True if embeddings exist, False otherwise
        """
        try:
            person_dir = self.output_dir / face_id
            embeddings_file = person_dir / 'embeddings.pt'
            visibility_file = person_dir / 'visibility.pt'
            pids_file = person_dir / 'pids.pt'

            return embeddings_file.exists() and visibility_file.exists() and pids_file.exists()

        except Exception as e:
            return False

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
        """
        GPU worker thread that processes face detection and person embedding jobs.

        Args:
            None

        Returns:
            None
        """
        # --- Construct models once, warmup once ---
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
            pass


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
                continue

            q_wait_ms = (time.time() - job.ts) * 1000.0
            with self.mode_lock:
                mode = self.mode

            if mode == Mode.FACES:
                # Process both FACE and PERSON jobs in FACES mode
                if job.kind == 'FACE' and job.camera_id == 'right':
                    # Face detection and recognition
                    t0 = time.time()
                    face_results = self.face_pipeline.process_frame(job.frame, thresh=0.5, input_size=(640, 640))
                    infer_ms = (time.time() - t0) * 1000.0
                    self._stats['face_qwait_ms'].append(q_wait_ms)
                    self._stats['face_infer_ms'].append(infer_ms)

                    recognized_faces = []
                    self.debug_counters['face_detections'] += len(face_results)

                    for face_data in face_results:
                        if face_data.get('embedding') is None:
                            continue

                        # Convert similarity threshold to distance threshold
                        distance_threshold = (1.0 / self.similarity_threshold) - 1.0

                        match, sim = self.similarity_search.find_face_match_euclidean(
                            face_data['embedding'], threshold=distance_threshold
                        )
                        if match:
                            self.debug_counters['recognized_faces'] += 1
                            recognized_faces.append((face_data, match))
                        else:
                            pass

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
                            if self.should_start_frame_collection(person_name, face_id):
                                # If same person or no one waiting, set/update waiting person
                                if self.waiting_person is None or self.waiting_person['name'] == person_name:
                                    if self.waiting_person is None:
                                        self.waiting_person = {
                                            'name': person_name,
                                            'face_id': face_id,
                                            'entered_zone_time': None,
                                            'camera_id': job.camera_id
                                        }
                                    # If same person, just refresh (keep existing state)
                                else:
                                    # Different person recognized while someone waiting - ignore for now
                                    pass

                elif job.kind == 'PERSON':
                    # Person detection to track zone entry (only for RIGHT camera in FACES mode)
                    # CENTER camera doesn't do zone checking in FACES mode
                    if job.camera_id != 'right':
                        continue  # Only RIGHT camera does zone tracking in FACES mode

                    t0 = time.time()
                    person_dets, _ = self.person_detector.detect_persons(job.frame)
                    infer_ms = (time.time() - t0) * 1000.0
                    self._stats['person_qwait_ms'].append(q_wait_ms)
                    self._stats['person_infer_ms'].append(infer_ms)


                    # Check if waiting person is in zone (RIGHT camera only)
                    with self.person_tracking_lock:
                        if self.waiting_person is not None and person_dets:
                            best_person = max(person_dets, key=lambda p: p['confidence'])
                            person_name = self.waiting_person['name']

                            if self.is_person_in_zone(best_person['bbox']):
                                # Person entered zone
                                if self.waiting_person['entered_zone_time'] is None:
                                    self.waiting_person['entered_zone_time'] = time.time()

                                # Check if countdown completed
                                elapsed = time.time() - self.waiting_person['entered_zone_time']
                                if elapsed >= self.countdown_duration:
                                    # Start collection!
                                    face_id = self.waiting_person['face_id']
                                    camera_id = self.waiting_person['camera_id']

                                    self.start_frame_collection(person_name, face_id, camera_id)

                                    # Switch mode to COLLECT
                                    with self.mode_lock:
                                        self.mode = Mode.COLLECT

                                    # Clear waiting person
                                    self.waiting_person = None
                                else:
                                    pass
                            else:
                                # Person left zone, reset countdown
                                if self.waiting_person['entered_zone_time'] is not None:
                                    self.waiting_person['entered_zone_time'] = None

            else:  # COLLECT
                if job.kind != 'PERSON':
                    continue

                # Detect persons in this frame
                t0 = time.time()
                person_dets, _ = self.person_detector.detect_persons(job.frame)
                infer_ms = (time.time() - t0) * 1000.0
                self._stats['person_qwait_ms'].append(q_wait_ms)
                self._stats['person_infer_ms'].append(infer_ms)


                if not person_dets or not self.active_collections:
                    if not person_dets:
                        pass
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

                    crop = self.person_detector.get_person_crop(job.frame, best['bbox'], padding=10)
                    if crop is None:
                        continue


                    # Convert BGR to RGB (GStreamer outputs BGR, but BPBreID expects RGB)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    t_emb_start = time.time()
                    emb, visibility = self.person_embedder.extract_embedding(crop_rgb)
                    emb_time = (time.time() - t_emb_start) * 1000.0

                    if emb is None or visibility is None:
                        continue


                    # Collect embeddings with augmentation on-the-fly
                    with self.frame_collection_lock:
                        if person_name in self.active_collections:  # Double-check still active
                            cdata = self.active_collections[person_name]
                            cam_data = cdata['cameras'][job.camera_id]

                            # Determine augmentation type based on frame count
                            frame_idx = cam_data['frames_collected']

                            # For each camera, collect 25 frames total:
                            # - 5 normal (ground truth)
                            # - 5 high brightness
                            # - 5 slight rotation
                            # - 5 noise
                            # - 5 additional variations

                            augmentation_type = None
                            if frame_idx < 5:
                                # First 5: Normal (ground truth)
                                augmentation_type = "normal"
                                augmented_crop = crop_rgb
                            elif frame_idx < 10:
                                # Next 5: High brightness (randomly select 5 from collected frames)
                                augmentation_type = "brightness"
                                augmented_crop = apply_brightness_augmentation(crop_rgb, brightness_factor=2.0)
                            elif frame_idx < 15:
                                # Next 5: Slight rotation
                                augmentation_type = "rotation"
                                augmented_crop = apply_rotation_augmentation(crop_rgb, max_angle=5)
                            elif frame_idx < 20:
                                # Next 5: Noise
                                augmentation_type = "noise"
                                augmented_crop = apply_noise_augmentation(crop_rgb, noise_level=15)
                            else:
                                # Last 5: Random mix
                                aug_choice = random.choice(['brightness', 'rotation', 'noise'])
                                if aug_choice == 'brightness':
                                    augmentation_type = "brightness_extra"
                                    augmented_crop = apply_brightness_augmentation(crop_rgb, brightness_factor=0.8)
                                elif aug_choice == 'rotation':
                                    augmentation_type = "rotation_extra"
                                    augmented_crop = apply_rotation_augmentation(crop_rgb, max_angle=10)
                                else:
                                    augmentation_type = "noise_extra"
                                    augmented_crop = apply_noise_augmentation(crop_rgb, noise_level=20)

                            # Extract embedding from augmented crop
                            t_aug_start = time.time()
                            aug_emb, aug_visibility = self.person_embedder.extract_embedding(augmented_crop)
                            aug_time = (time.time() - t_aug_start) * 1000.0

                            if aug_emb is None or aug_visibility is None:
                                continue


                            # Append augmented embedding with unsqueeze to add batch dimension
                            cam_data['embeddings'].append(aug_emb.unsqueeze(0))
                            cam_data['visibility'].append(aug_visibility.unsqueeze(0))
                            cam_data['frames_collected'] += 1
                            cam_data['last_person_box'] = best['bbox']
                            cam_data['last_augmentation'] = augmentation_type  # For display

                            cdata['total_frames_collected'] = sum(
                                v['frames_collected'] for v in cdata['cameras'].values()
                            )
                            self.debug_counters['embedding_successes'] += 1


                            # If all cameras hit 25 frames, complete
                            done = all(v['frames_collected'] >= 25 for v in cdata['cameras'].values())

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

                    # Store augmented crop for visualization (convert back to BGR for display)
                    camera_data['augmented_crop'] = cv2.cvtColor(augmented_crop, cv2.COLOR_RGB2BGR)
                    camera_data['augmentation_type'] = augmentation_type

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

            try:
                self.create_camera_pipeline(camera_id, config)
                # GStreamer documentation: Allow pipeline to stabilize before next
                time.sleep(0.5)
            except Exception as e:
                pass

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
            pass

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
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.cameras[camera_id]['pipeline']:
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    self.cameras[camera_id]['playing'] = True

        return True

    def should_start_frame_collection(self, person_name, face_id):
        """
        Check if frame collection should start for a person.

        Args:
            person_name (str): Name of the person
            face_id (str): Firebase user ID

        Returns:
            bool: True if collection should start, False otherwise
        """
        # Check if embeddings already exist for this person
        if self.check_person_embeddings_exist(face_id):
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
            face_id (str): Firebase user ID from firebase_users table
            trigger_camera_id (str): Camera that triggered the collection

        Returns:
            bool: True if collection started, False otherwise
        """
        with self.frame_collection_lock:
            if person_name not in self.active_collections:
                self.active_collections[person_name] = {
                    'face_id': face_id,  # Store face_id for reference
                    'trigger_camera': trigger_camera_id,
                    'start_time': time.time(),
                    'cameras': {
                        'center': {'frames_collected': 0, 'embeddings': [], 'visibility': [], 'last_person_box': None, 'last_augmentation': None},
                        'right': {'frames_collected': 0, 'embeddings': [], 'visibility': [], 'last_person_box': None, 'last_augmentation': None}
                    },
                    'total_frames_collected': 0
                }
                return True
        return False

    def complete_frame_collection(self, person_name):
        """
        Finalize a collection, save embeddings to .pt files, and clean up.

        Args:
            person_name (str): Name of the person

        Returns:
            None
        """
        if person_name not in self.active_collections:
            return

        collection_data = self.active_collections[person_name]
        face_id = collection_data['face_id']

        # Collect all embeddings and visibility from all cameras
        all_embeddings_list = []
        all_visibility_list = []

        for cam, cam_data in collection_data['cameras'].items():
            if cam_data['embeddings']:
                all_embeddings_list.extend(cam_data['embeddings'])
                all_visibility_list.extend(cam_data['visibility'])

        # Save all collected embeddings to .pt file with face_id as PID
        if all_embeddings_list and all_visibility_list:
            # Aggregate ONCE
            all_embeddings = torch.cat(all_embeddings_list, dim=0)
            all_visibility = torch.cat(all_visibility_list, dim=0)

            # PRIMARY: save to memory (use firebase user ID directly as gallery key)
            self.live_gallery.add_person(
                pid=face_id,
                embeddings=all_embeddings,
                visibility=all_visibility
            )
            self.save_person_embeddings_to_pt(person_name, all_embeddings_list, all_visibility_list, face_id)

        # Remove from active collections
        del self.active_collections[person_name]

        # Update debug counters
        total_frames = collection_data['total_frames_collected']

        camera_summary = ', '.join([
            f"{cam}: {cam_data['frames_collected']}"
            for cam, cam_data in collection_data['cameras'].items()
            if cam_data['frames_collected'] > 0
        ])


    def cleanup_stale_collections(self):
        """
        Clean up collections that have been running too long and waiting persons that timed out.

        Args:
            None

        Returns:
            None
        """
        current_time = time.time()
        stale_collections = []

        # Cleanup stale active collections
        with self.frame_collection_lock:
            for person_name, collection_data in self.active_collections.items():
                # If collection is running for more than 30 seconds, consider it stale
                if current_time - collection_data['start_time'] > 30:
                    stale_collections.append(person_name)

            for person_name in stale_collections:
                # Check if we have any embeddings collected for this person
                collection_data = self.active_collections.get(person_name)
                if collection_data:
                    total_collected = collection_data['total_frames_collected']
                    if total_collected > 0:
                        # Save whatever we've collected
                        self.complete_frame_collection(person_name)
                    else:
                        # Nothing collected, just remove
                        del self.active_collections[person_name]

        # Cleanup waiting person if they haven't entered zone within 10 seconds
        with self.person_tracking_lock:
            if self.waiting_person is not None:
                # If person hasn't entered zone yet, track from recognition time
                # We'll use a simple heuristic: if no entered_zone_time after 10 seconds, clear
                # This is a simplified approach - in production you might want to track recognition_time
                pass  # For now, we'll rely on manual clearing when new person is recognized

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

                    # Get current mode and waiting person status
                    with self.mode_lock:
                        mode = self.mode

                    should_detect = (frame_count % self.detection_interval == 0)

                    if mode == Mode.FACES:
                        # RIGHT camera sends FACE jobs for face recognition
                        if camera_id == 'right' and should_detect:
                            try:
                                self.job_q.put_nowait(Job('FACE', camera_id, frame, time.time()))
                            except:
                                pass  # drop if queue full

                        # Only RIGHT camera sends PERSON jobs for zone tracking in FACES mode
                        if camera_id == 'right':
                            with self.person_tracking_lock:
                                has_waiting_person = self.waiting_person is not None

                            if has_waiting_person and should_detect:
                                try:
                                    self.job_q.put_nowait(Job('PERSON', camera_id, frame, time.time()))
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
                    # Show augmentation type if available
                    aug_type = ""
                    if person_name in self.active_collections:
                        cam_data = self.active_collections[person_name]['cameras'].get(camera_id, {})
                        last_aug = cam_data.get('last_augmentation', None)
                        if last_aug:
                            aug_type = f" [{last_aug.upper()}]"
                    embedding_status = f"BPBREID{aug_type}"
                    color = (255, 165, 0)  # Orange for BPBreID person detection
                    thickness = 4  # Thicker for visibility
                    label = f"{person_name} {embedding_status}"
                else:
                    # Face detection box
                    if person_name in self.active_collections:
                        embedding_status = f"COLLECTING [PT]"
                        color = (0, 255, 255)  # Cyan when actively collecting
                    else:
                        embedding_status = f"RECOGNIZED [PT]"
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

        # Draw designated zone polygon ONLY on RIGHT camera (scaled to display coordinates)
        if camera_id == 'right':
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
        camera_name = camera_data['config'].get('name', f'Camera {camera_id.title()}')
        if camera_id == 'right':
            camera_role = "PRIMARY"
        else:
            camera_role = "SECONDARY"

        cv2.putText(display_frame, f"{camera_name} ({camera_role})", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Add GPU-safe status
        status_color = (0, 255, 0) if self.collecting_embeddings else (0, 0, 255)
        status_text = f"Collection: {'ON' if self.collecting_embeddings else 'OFF'} [PT FILES]"
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
                cv2.putText(display_frame, wait_text, (10, 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, wait_color, 2)

        # Add FPS
        fps_text = f"FPS: {self.current_fps[camera_id]}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display augmented crop preview if available (bottom-right corner)
        if 'augmented_crop' in camera_data and camera_data['augmented_crop'] is not None:
            aug_crop = camera_data['augmented_crop']
            aug_type = camera_data.get('augmentation_type', 'unknown')

            # Resize crop to fit in preview (max 200x200)
            crop_h, crop_w = aug_crop.shape[:2]
            scale = min(200 / crop_w, 200 / crop_h)
            preview_w = int(crop_w * scale)
            preview_h = int(crop_h * scale)
            aug_preview = cv2.resize(aug_crop, (preview_w, preview_h))

            # Position in bottom-right corner
            preview_x = display_frame.shape[1] - preview_w - 10
            preview_y = display_frame.shape[0] - preview_h - 50

            # Draw border and overlay
            cv2.rectangle(display_frame,
                         (preview_x - 2, preview_y - 2),
                         (preview_x + preview_w + 2, preview_y + preview_h + 2),
                         (255, 165, 0), 2)
            display_frame[preview_y:preview_y+preview_h, preview_x:preview_x+preview_w] = aug_preview

            # Add augmentation label above preview
            aug_label = f"AUG: {aug_type.upper()}"
            label_size = cv2.getTextSize(aug_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame,
                         (preview_x, preview_y - label_size[1] - 10),
                         (preview_x + label_size[0], preview_y - 2),
                         (255, 165, 0), -1)
            cv2.putText(display_frame, aug_label,
                       (preview_x, preview_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        cv2.putText(canvas, "BPBreID Person Embedding Collection (.PT FILES):",
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
                total_collected = collection_data['total_frames_collected']
                elapsed = time.time() - collection_data['start_time']

                # Get augmentation types for display
                center_aug = collection_data['cameras']['center'].get('last_augmentation', 'N/A')
                right_aug = collection_data['cameras']['right'].get('last_augmentation', 'N/A')

                # Target is 25 per camera * 2 cameras = 50 (with augmentations)
                active_text = f"  {person_name}: {total_collected}/50 [C:{center_frames}/25 ({center_aug}), R:{right_frames}/25 ({right_aug})] ({elapsed:.1f}s)"
                cv2.putText(canvas, active_text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
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
                pipeline = self.cameras[camera_id]['pipeline']
                ret = pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    pass
                # GStreamer documentation: serialize pipeline startup
                time.sleep(1.0)
            except Exception as e:
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
                    cv2.imshow('Person Embedding Collection (.PT Files)', display)

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

        # Give threads time to see running=False
        time.sleep(0.2)

        # Complete any active collections
        with self.frame_collection_lock:
            if self.active_collections:
                for person_name in list(self.active_collections.keys()):
                    self.complete_frame_collection(person_name)

        # Wait for GPU worker thread to finish
        if self.gpu_worker_thread and self.gpu_worker_thread.is_alive():
            self.gpu_worker_thread.join(timeout=3.0)
            if self.gpu_worker_thread.is_alive():
                pass

        # Close Redis connection
        if self.redis_client is not None:
            try:
                self.redis_client.close()
            except Exception as e:
                pass

        # Stop camera pipelines
        for camera_id in ['center', 'right']:
            try:
                pipeline = self.cameras[camera_id]['pipeline']
                if pipeline:
                    pipeline.set_state(Gst.State.NULL)

                thread = self.cameras[camera_id].get('thread')
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        pass
            except Exception as e:
                pass

        cv2.destroyAllWindows()
