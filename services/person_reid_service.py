import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ByteTrack'))
import json
import numpy as np
import cv2
import torch
import threading
import time
from queue import Queue
from collections import namedtuple
import traceback
from datetime import datetime
from pathlib import Path

# ByteTracker imports
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

# Import camera management
from camera import CameraManager, create_camera_configs_from_ips

# Import our utility modules
from utils.database_manager import DatabaseManager
from utils.person_detector import PersonDetector

# BPBreID imports
from torchreid.scripts.main import build_config
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.utils.constants import bn_correspondants


class SimpleArgs:
    """Minimal args for build_config."""
    def __init__(self):
        self.root = ''
        self.save_dir = 'log'
        self.job_id = 'inference'
        self.inference_enabled = False
        self.sources = None
        self.targets = None
        self.transforms = None
        self.opts = []


class PersonReIDService:
    """
    Person Re-Identification Service.
    
    This service:
    1. Uses collected body embeddings from the database
    2. Tracks persons across multiple cameras
    3. Identifies persons using similarity search against stored embeddings
    4. Provides real-time person tracking and identification
    """
    
    def __init__(self, stream_configs, config_path, weights_path, gallery_dir=None,
                 voting_threshold=4.5, voting_window=30, min_votes=15,
                 matching_threshold=6.0, device='cuda'):
        """
        Initialize Dynamic Camera Person ReID Service with BPBreID.

        Args:
            stream_configs (list): List of camera stream configurations
            config_path (str): Path to BPBreID config YAML
            weights_path (str): Path to BPBreID model weights
            gallery_dir (str): Directory with gallery embeddings (optional)
            voting_threshold (float): Distance threshold for counting a vote (default 4.5)
            voting_window (int): Number of frames to collect votes (default 30)
            min_votes (int): Minimum votes needed to cache an ID (default 15)
            matching_threshold (float): Distance threshold for matching/display (default 6.0)
            device (str): 'cuda' or 'cpu'

        Raises:
            ValueError: If no cameras are provided
        """
        self.num_cameras = len(stream_configs)
        if self.num_cameras == 0:
            raise ValueError("At least one camera is required")

        self.stream_configs = stream_configs
        self.voting_threshold = voting_threshold
        self.voting_window = voting_window
        self.min_votes = min_votes
        self.matching_threshold = matching_threshold
        self.device = device
        self.last_detection_time = {}
        self.reset_delay = 4  # seconds

        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_list_log_file = os.path.join(self.logs_dir, f"master_list_{timestamp}.txt")

        # Create log file with simple header
        with open(self.master_list_log_file, 'w') as f:
            f.write(f"PERSON RE-ID LOG - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

        print(f"Log file created: {self.master_list_log_file}")
        self.logged_tracks = set()

        # Initialize utility modules
        print("Initializing database manager...")
        self.db_manager = DatabaseManager()

        print("Initializing person detector...")
        self.person_detector = PersonDetector()

        # Initialize BPBreID model
        print("\n[INIT] Loading BPBreID model...")
        self._initialize_reid(config_path, weights_path, gallery_dir, device)

        # Voting-based cache structures (replacing old cache)
        self.track_voting = {}  # {(camera_id, track_id): {'votes': {}, 'frame_count': 0, 'distances': {}}}
        self.cached_tracks = {}  # {(camera_id, track_id): {'person_id': str, 'distance': float, 'votes': int}}

        # Database embedding polling
        self.last_embedding_check = 0
        self.embedding_check_interval = 5  # Check every 5 seconds

        # Load all existing CONSUMED embeddings on startup
        print("\n[INIT] Loading existing embeddings from database...")
        self._load_all_consumed_embeddings()

        # Create tracker arguments
        tracker_args = type('Args', (object,), {
            'track_thresh': 0.5,
            'match_thresh': 0.8,
            'buffer_size': 30,
            'mot20': False,
            'aspect_ratio_thresh': 1.6,
            'min_box_area': 10,
            'frame_rate': 30,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        })
        
        # Create trackers for each camera dynamically
        self.trackers = {}
        for i, config in enumerate(stream_configs):
            camera_id = f"cam_{i+1}"
            self.trackers[camera_id] = BYTETracker(tracker_args)
        
        # Display configuration
        self.camera_width = 640
        self.camera_height = 640
        # Calculate display dimensions based on number of cameras
        self.grid_cols = int(np.ceil(np.sqrt(self.num_cameras)))
        self.grid_rows = int(np.ceil(self.num_cameras / self.grid_cols))
        self.display_width = self.grid_cols * self.camera_width
        self.display_height = self.grid_rows * self.camera_height
        
        # Processing configuration
        self.detection_interval = 1
        
        # Initialize camera manager
        self.camera_manager = CameraManager(self.camera_width, self.camera_height)
        
        # Add cameras to manager
        for i, config in enumerate(stream_configs):
            camera_id = f"cam_{i+1}"
            self.camera_manager.add_camera(camera_id, config)
        
        # Control flags
        self.running = False
        
        # Frame synchronization
        self.display_lock = threading.Lock()
        self.stable_display_buffer = {}
        for i in range(self.num_cameras):
            camera_id = f"cam_{i+1}"
            self.stable_display_buffer[camera_id] = None
        self.stable_display_buffer['timestamp'] = time.time()
        
        # Database statistics (initialize before database connection)
        self.db_stats = {
            'total_people': 0,
            'total_embeddings': 0,
            'last_updated': time.time()
        }
        
        # Performance tracking - Dynamic for all cameras
        self.fps_counters = {}
        self.fps_start_times = {}
        self.current_fps = {}
        for i in range(self.num_cameras):
            camera_id = f"cam_{i+1}"
            self.fps_counters[camera_id] = 0
            self.fps_start_times[camera_id] = time.time()
            self.current_fps[camera_id] = 0
        
        # Initialize database connection first
        self.initialize_database_connection()
        self.master_person_list = []

    def _initialize_reid(self, config_path, weights_path, gallery_dir, device):
        """Initialize BPBreID model and gallery."""
        # Build config and extractor
        dummy_args = SimpleArgs()
        self.cfg = build_config(args=dummy_args, config_file=config_path)
        self.cfg.use_gpu = (device.startswith('cuda') and torch.cuda.is_available())

        model_path = weights_path or self.cfg.model.load_weights
        self.extractor = FeatureExtractor(
            self.cfg,
            model_path=model_path,
            device=device if torch.cuda.is_available() else 'cpu',
            num_classes=1,
            verbose=True
        )
        self.device = self.extractor.device

        # Load gallery (optional - can start with empty gallery)
        print("\n[INIT] Loading gallery...")
        if gallery_dir and Path(gallery_dir).exists():
            gallery_path = Path(gallery_dir)
            embeddings_file = gallery_path / 'gallery_embeddings.pt'

            if embeddings_file.exists():
                self.gallery_embeddings = torch.load(embeddings_file, map_location=self.device)
                self.gallery_visibility = torch.load(gallery_path / 'gallery_visibility.pt', map_location=self.device)
                self.gallery_pids = torch.load(gallery_path / 'gallery_pids.pt', map_location=self.device)
                print(f"        Gallery: {self.gallery_embeddings.shape[0]} samples, "
                      f"{self.gallery_embeddings.shape[1]} parts, PIDs: {torch.unique(self.gallery_pids).tolist()}")
            else:
                print("        No initial gallery found, starting with empty gallery")
                self._initialize_empty_gallery()
        else:
            print("        No gallery directory provided, starting with empty gallery")
            self._initialize_empty_gallery()

    def _initialize_empty_gallery(self):
        """Initialize empty gallery tensors."""
        # Create empty tensors with proper shape: [0, num_parts, embedding_dim]
        # Assuming 6 body parts and 512-dim embeddings (BPBreID default)
        self.gallery_embeddings = torch.empty(0, 6, 512, device=self.device)
        self.gallery_visibility = torch.empty(0, 6, device=self.device)
        self.gallery_pids = torch.empty(0, dtype=torch.long, device=self.device)
        print("        Initialized empty gallery - will load embeddings dynamically from database")

    def _load_all_consumed_embeddings(self):
        """Load all CONSUMED embeddings from database on startup."""
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                print("        No database connection, skipping initial embedding load")
                return

            cur = conn.cursor()
            cur.execute("""
                SELECT person_id, person_name, pt_path, created_at
                FROM embedding_status
                WHERE status = 'CONSUMED'
                ORDER BY created_at ASC
            """)

            results = cur.fetchall()
            cur.close()
            conn.close()

            if not results:
                print("        No CONSUMED embeddings found in database")
                return

            print(f"        Found {len(results)} CONSUMED embedding(s) to load")

            # Load each embedding
            for row in results:
                person_id = row[0]
                person_name = row[1]
                pt_path = row[2]

                print(f"        Loading {person_name} (person_id={person_id})...")

                embeddings, visibility, pids = self.load_person_embeddings(pt_path)

                if embeddings is not None:
                    self.add_to_gallery(embeddings, visibility, pids)
                else:
                    print(f"        ✗ Failed to load embeddings for {person_name}")

            print(f"        ✓ Loaded {len(results)} person embeddings into gallery\n")

        except Exception as e:
            print(f"[ERROR] Failed to load consumed embeddings: {e}")
            import traceback
            traceback.print_exc()

    def initialize_database_connection(self):
        """
        Initialize and verify database connection.

        Establishes connection to the database and retrieves initial statistics
        for person embeddings and stored people.

        Raises:
            Exception: If database initialization fails
        """
        print("🔍 Checking database connection...")
        try:
            stats = self.db_manager.get_database_stats()

            self.db_stats['total_people'] = stats['person_people']
            self.db_stats['total_embeddings'] = stats['person_embeddings']
            self.db_stats['last_updated'] = time.time()

            print(f"Database connection successful!")

        except Exception as e:
            print(f"Database initialization failed: {e}")
            raise

    def check_for_new_embeddings(self):
        """Check embedding_status table for new READY embeddings."""
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                return []

            cur = conn.cursor()
            cur.execute("""
                SELECT person_id, person_name, pt_path, created_at
                FROM embedding_status
                WHERE status = 'READY'
                ORDER BY created_at ASC
            """)

            results = cur.fetchall()
            cur.close()
            conn.close()

            new_embeddings = []
            for row in results:
                new_embeddings.append({
                    'person_id': row[0],
                    'person_name': row[1],
                    'pt_path': row[2],
                    'created_at': row[3]
                })

            return new_embeddings

        except Exception as e:
            print(f"[DB_ERROR] Failed to check for new embeddings: {e}")
            return []

    def load_person_embeddings(self, pt_path):
        """Load person embeddings from .pt files."""
        try:
            pt_dir = Path(pt_path)
            embeddings_file = pt_dir / 'embeddings.pt'
            visibility_file = pt_dir / 'visibility.pt'
            pids_file = pt_dir / 'pids.pt'

            if not (embeddings_file.exists() and visibility_file.exists() and pids_file.exists()):
                print(f"[LOAD_ERROR] Missing .pt files in {pt_dir}")
                return None, None, None

            embeddings = torch.load(embeddings_file, map_location=self.device)
            visibility = torch.load(visibility_file, map_location=self.device)
            pids = torch.load(pids_file, map_location=self.device)

            print(f"[LOAD] Loaded embeddings from {pt_dir}")
            print(f"       Shape: {embeddings.shape}, PIDs: {torch.unique(pids).tolist()}")

            return embeddings, visibility, pids

        except Exception as e:
            print(f"[LOAD_ERROR] Failed to load embeddings from {pt_path}: {e}")
            return None, None, None

    def update_embedding_status(self, person_id, status='CONSUMED'):
        """Update embedding status in database."""
        try:
            current_timestamp = int(time.time() * 1000)  # milliseconds

            conn = self.db_manager.get_connection()
            if not conn:
                return False

            cur = conn.cursor()
            cur.execute("""
                UPDATE embedding_status
                SET status = %s, updated_at = %s
                WHERE person_id = %s
            """, (status, current_timestamp, person_id))

            conn.commit()
            cur.close()
            conn.close()

            print(f"[DB] Updated person_id={person_id} to status={status}")
            return True

        except Exception as e:
            print(f"[DB_ERROR] Failed to update embedding status: {e}")
            return False

    def add_to_gallery(self, embeddings, visibility, pids):
        """Add new embeddings to the gallery dynamically."""
        try:
            # Append to existing gallery
            self.gallery_embeddings = torch.cat([self.gallery_embeddings, embeddings], dim=0)
            self.gallery_visibility = torch.cat([self.gallery_visibility, visibility], dim=0)
            self.gallery_pids = torch.cat([self.gallery_pids, pids], dim=0)

            print(f"[GALLERY] Added {embeddings.shape[0]} new samples")
            print(f"          Total gallery size: {self.gallery_embeddings.shape[0]} samples")
            print(f"          Total unique PIDs: {len(torch.unique(self.gallery_pids))}")

            return True

        except Exception as e:
            print(f"[GALLERY_ERROR] Failed to add embeddings to gallery: {e}")
            return False

    def poll_and_load_new_embeddings(self):
        """Poll database for new embeddings and load them into gallery."""
        current_time = time.time()

        # Only check at intervals
        if current_time - self.last_embedding_check < self.embedding_check_interval:
            return

        self.last_embedding_check = current_time

        # Check for new embeddings
        new_embeddings = self.check_for_new_embeddings()

        if not new_embeddings:
            return

        print(f"\n[POLL] Found {len(new_embeddings)} new embedding(s) with READY status")

        # Load and add each new embedding
        for emb_info in new_embeddings:
            person_id = emb_info['person_id']
            person_name = emb_info['person_name']
            pt_path = emb_info['pt_path']

            print(f"[POLL] Loading embeddings for {person_name} (person_id={person_id})...")

            embeddings, visibility, pids = self.load_person_embeddings(pt_path)

            if embeddings is not None:
                # Add to gallery
                if self.add_to_gallery(embeddings, visibility, pids):
                    # Mark as consumed
                    self.update_embedding_status(person_id, status='CONSUMED')
                    print(f"[POLL] ✓ Successfully loaded and marked {person_name} as CONSUMED\n")
                else:
                    print(f"[POLL] ✗ Failed to add {person_name} to gallery\n")
            else:
                print(f"[POLL] ✗ Failed to load embeddings for {person_name}\n")

    def extract_test_embeddings(self, model_output):
        """Extract embeddings from BPBreID model output."""
        embeddings_dict, visibility_dict, _, _, _, _ = model_output

        embeddings_list = []
        visibility_list = []

        for test_emb in self.cfg.model.bpbreid.test_embeddings:
            embds = embeddings_dict[test_emb]
            embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))

            vis_key = test_emb if test_emb not in bn_correspondants else bn_correspondants[test_emb]
            vis_scores = visibility_dict[vis_key]
            visibility_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))

        embeddings = torch.cat(embeddings_list, dim=1)
        visibility = torch.cat(visibility_list, dim=1)
        return embeddings, visibility

    def prepare_detections_for_tracker(self, person_boxes_track):
        """
        Prepare detections in the correct format for BYTETracker.

        Converts person detection boxes to the format expected by BYTETracker,
        including filtering by area and aspect ratio constraints.

        Args:
            person_boxes_track (list): List of detections in format [x1, y1, x2, y2, score]

        Returns:
            torch.Tensor: Tensor with shape (N, 6) formatted for BYTETracker [x1, y1, x2, y2, score, class]
        """
        if not person_boxes_track:
            return torch.tensor([], dtype=torch.float32).reshape(0, 6)
        
        # Convert to numpy array first
        detections_array = np.array(person_boxes_track)
        
        # Check if we have the right shape
        if detections_array.ndim != 2 or detections_array.shape[1] != 5:
            return torch.tensor([], dtype=torch.float32).reshape(0, 6)
        
        # Convert to format expected by BYTETracker [x1, y1, x2, y2, score, class]
        detections_list = []
        for box in person_boxes_track:
            if len(box) >= 5:
                x1, y1, x2, y2, score = box[:5]
                w = x2 - x1
                h = y2 - y1
                
                # Filter by area and aspect ratio
                if w > 0 and h > 0 and w * h > 100 and w / h <= 2.0:
                    detection = [float(x1), float(y1), float(x2), float(y2), float(score), 1.0]
                    detections_list.append(detection)
        
        if not detections_list:
            return torch.tensor([], dtype=torch.float32).reshape(0, 6)
        
        return torch.tensor(detections_list, dtype=torch.float32)
    


    def cache_master_list(self, person_name, camera_id, track_id):
        """
        Append cached person to master list and save to database once per unique track.

        Saves person identification to both master list and database, ensuring
        each unique track is only logged once.

        Args:
            person_name (str): Identified person name
            camera_id (str): Camera identifier
            track_id (int): Track identifier
        """
        if person_name != "Unknown":
            # Create unique key for this track
            track_key = (camera_id, track_id, person_name)
            
            # Only save if we haven't saved this track before
            if track_key not in self.logged_tracks:
                # Add to master list
                entry = {
                    'person_name': person_name,
                    'camera_id': camera_id, 
                    'track_id': track_id,
                    'timestamp': time.time()
                }
                self.master_person_list.append(entry)
                
                # Insert into database only
                try:
                    db_success = self.db_manager.insert_person_reid_log(person_name, camera_id, track_id)
                    
                    if db_success:
                        print(f"[DB SAVED] {person_name} detected on {camera_id} with track {track_id}")
                    else:
                        print(f"[DB FAILED] {person_name} detected on {camera_id} with track {track_id}")
                    
                    # Mark this track as saved
                    self.logged_tracks.add(track_key)
                    
                except Exception as e:
                    print(f"Error saving to database: {e}")

    def process_frame_for_reid(self, frame, camera_id):
        """
        Process frame for person re-identification with BPBreID and voting-based caching.

        Args:
            frame (numpy.ndarray): Input video frame
            camera_id (str): Camera identifier

        Returns:
            list: List of person re-identification results with tracking and match information
        """
        try:
            # Poll for new embeddings periodically
            self.poll_and_load_new_embeddings()

            # Detect persons using person detector utility
            person_detections, person_boxes_track = self.person_detector.detect_persons(frame)

            now = time.time()
            sample_id = int(camera_id.split('_')[1]) - 1

            if len(person_boxes_track) > 0:
                self.last_detection_time[sample_id] = now
            else:
                last_seen = self.last_detection_time.get(sample_id, now)
                if now - last_seen > 0.1:
                    tracker = self.trackers[camera_id]
                    empty_outputs = torch.tensor([], dtype=torch.float32).reshape(0, 6)

                    if torch.cuda.is_available():
                        empty_outputs = empty_outputs.cuda()

                    try:
                        online_targets = tracker.update(
                            empty_outputs,
                            [frame.shape[0], frame.shape[1]],
                            (frame.shape[0], frame.shape[1])
                        )

                        # Clear voting and cached tracks for this camera
                        keys_to_remove = [key for key in self.cached_tracks.keys() if key[0] == camera_id]
                        for key in keys_to_remove:
                            del self.cached_tracks[key]

                        keys_to_remove = [key for key in self.track_voting.keys() if key[0] == camera_id]
                        for key in keys_to_remove:
                            del self.track_voting[key]

                    except Exception as e:
                        print(f"Error updating tracker with empty values: {e}")

                    self.last_detection_time[sample_id] = now

            if not person_detections or not person_boxes_track:
                return []

            target_box = []
            reid_results = []

            # Prepare detections for tracking
            outputs = self.prepare_detections_for_tracker(person_boxes_track)

            if torch.cuda.is_available() and hasattr(self.trackers[camera_id], 'args'):
                device = getattr(self.trackers[camera_id].args, 'device', 'cpu')
                if hasattr(device, 'type'):
                    device_str = device.type
                else:
                    device_str = str(device)

                if 'cuda' in device_str:
                    outputs = outputs.cuda()

            try:
                if outputs is not None and len(outputs) > 0:
                    online_targets = self.trackers[camera_id].update(
                        outputs,
                        [frame.shape[0], frame.shape[1]],
                        (frame.shape[0], frame.shape[1])
                    )

                    for target in online_targets:
                        target_box.append([target.track_id, target.tlwh[0], target.tlwh[1],
                                        target.tlwh[2], target.tlwh[3], target.track_id, camera_id])

                else:
                    online_targets = []
            except Exception as e:
                traceback.print_exc()
                return []

            # Process each target for ReID with BPBreID and voting-based caching
            for target in target_box:
                track_id, x, y, w, h, _, cam_id = target
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                cache_key = (cam_id, track_id)

                # Check if this track is already cached
                if cache_key in self.cached_tracks:
                    person_name = self.cached_tracks[cache_key]['person_id']
                    result_distance = self.cached_tracks[cache_key]['distance']
                    is_cached = True

                else:
                    # Crop for ReID
                    person_crop = self.person_detector.get_person_crop(frame, [x1, y1, x2, y2], padding=10)

                    if person_crop is None:
                        continue

                    # Convert to RGB
                    crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                    # Extract embeddings using BPBreID
                    model_output = self.extractor([crop_rgb])
                    embeddings, visibility = self.extract_test_embeddings(model_output)
                    embeddings = embeddings.to(self.device)
                    visibility = visibility.to(self.device)

                    # Check if gallery is empty
                    if self.gallery_embeddings.shape[0] == 0:
                        person_name = "UNKNOWN"
                        result_distance = 999.0
                        is_cached = False
                    else:
                        # Compute distances using BPBreID distance calculation
                        with torch.no_grad():
                            distmat, _ = compute_distance_matrix_using_bp_features(
                                embeddings,
                                self.gallery_embeddings,
                                visibility,
                                self.gallery_visibility,
                                dist_combine_strat=self.cfg.test.part_based.dist_combine_strat,
                                batch_size_pairwise_dist_matrix=self.cfg.test.batch_size_pairwise_dist_matrix,
                                use_gpu=self.cfg.use_gpu,
                                metric='euclidean'
                            )

                        # Get best match
                        distances = distmat[0].cpu().numpy()
                        best_idx = np.argmin(distances)
                        best_pid = int(self.gallery_pids[best_idx].item())
                        best_dist = distances[best_idx]

                        person_name = str(best_pid)
                        result_distance = best_dist
                        is_cached = False

                        # Voting logic
                        if cache_key not in self.track_voting:
                            self.track_voting[cache_key] = {
                                'votes': {},
                                'frame_count': 0,
                                'distances': {}
                            }

                        voting_entry = self.track_voting[cache_key]
                        voting_entry['frame_count'] += 1

                        # Count vote if distance < voting_threshold
                        if best_dist < self.voting_threshold:
                            if person_name not in voting_entry['votes']:
                                voting_entry['votes'][person_name] = 0
                                voting_entry['distances'][person_name] = best_dist
                            voting_entry['votes'][person_name] += 1
                            voting_entry['distances'][person_name] = min(
                                voting_entry['distances'][person_name],
                                best_dist
                            )

                        # Check if we've collected enough frames
                        if voting_entry['frame_count'] >= self.voting_window:
                            if voting_entry['votes']:
                                max_votes_id = max(voting_entry['votes'], key=voting_entry['votes'].get)
                                max_votes = voting_entry['votes'][max_votes_id]

                                if max_votes >= self.min_votes:
                                    self.cached_tracks[cache_key] = {
                                        'person_id': max_votes_id,
                                        'distance': voting_entry['distances'][max_votes_id],
                                        'votes': max_votes
                                    }
                                    print(f"[CACHE] {cam_id} T{track_id} -> ID{max_votes_id} "
                                          f"(votes={max_votes}/{self.voting_window})")

                                    person_name = max_votes_id
                                    result_distance = voting_entry['distances'][max_votes_id]
                                    is_cached = True

                            del self.track_voting[cache_key]

                # Determine if recognized
                is_recognized = result_distance < self.matching_threshold

                reid_results.append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'person_name': person_name if is_recognized else "UNKNOWN",
                    'distance': result_distance,
                    'camera_id': cam_id,
                    'cached': is_cached,
                    'match': {'person_name': person_name, 'source_camera': cam_id} if is_recognized else None,
                    'similarity': 1.0 / (1.0 + result_distance)  # Convert distance to similarity for compatibility
                })

            # Cleanup disappeared tracks
            active_track_ids = {(camera_id, target[0]) for target in target_box}
            disappeared_tracks = set(self.cached_tracks.keys()) - active_track_ids
            for key in disappeared_tracks:
                if key[0] == camera_id:
                    del self.cached_tracks[key]

            disappeared_voting = set(self.track_voting.keys()) - active_track_ids
            for key in disappeared_voting:
                if key[0] == camera_id:
                    del self.track_voting[key]

            return reid_results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return []


    def get_cache_info(self):
        """
        Get information about the current BPBreID voting cache state.

        Returns detailed information about all cached tracks from voting system.

        Returns:
            dict: Dictionary containing cache information organized by camera
        """
        cache_info = {}
        for (camera_id, track_id), data in self.cached_tracks.items():
            if camera_id not in cache_info:
                cache_info[camera_id] = []
            cache_info[camera_id].append({
                'track_id': track_id,
                'person_id': data['person_id'],
                'distance': data['distance'],
                'votes': data['votes']
            })
        return cache_info

    def update_database_stats(self):
        """
        Update database statistics periodically.

        Refreshes database statistics including total people and embeddings
        count every 30 seconds for monitoring purposes.
        """
        try:
            current_time = time.time()
            # Update stats every 30 seconds
            if current_time - self.db_stats['last_updated'] > 30:
                stats = self.db_manager.get_database_stats()
                self.db_stats['total_people'] = stats['person_people']
                self.db_stats['total_embeddings'] = stats['person_embeddings']
                self.db_stats['last_updated'] = current_time
        except Exception as e:
            print(f"[WARNING] Failed to update database stats: {e}")

    def processing_thread(self, camera_id):
        """
        Main processing thread for each camera.

        Runs continuously to process frames from a specific camera,
        perform person detection and re-identification, and update display.

        Args:
            camera_id (str): Camera identifier
        """
        print(f"Starting processing thread for {camera_id} camera")
        
        # Initialize persistent detection data for this camera
        persistent_detections = []
        detection_timestamp = 0
        
        while self.running:
            try:
                camera = self.camera_manager.get_camera(camera_id)
                if not camera:
                    time.sleep(0.1)
                    continue
                
                frame = camera.get_frame()
                                
                if frame is not None:
                    frame_count = camera.get_frame_count()
                    
                    # Process detection every N frames
                    should_detect = (frame_count % self.detection_interval == 0)
                    
                    if should_detect:
                        # Process frame for person re-identification
                        reid_results = self.process_frame_for_reid(frame, camera_id)
                        
                        # Update persistent data
                        persistent_detections = reid_results
                        detection_timestamp = time.time()
                        
                        # Update database stats periodically
                        self.update_database_stats()
                    
                    # Always update the processed frame
                    current_time = time.time()
                    if (current_time - detection_timestamp) < 2.0:
                        processed_frame = self.draw_reid_results(
                            frame, persistent_detections, camera_id
                        )
                    else:
                        processed_frame = self.draw_reid_results(frame, [], camera_id)
                    
                    # Store processed frame
                    camera.set_processed_frame(processed_frame)
                
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error in processing thread for {camera_id}: {e}")
                time.sleep(0.1)
    
    def calculate_display_dimensions(self, original_width, original_height, target_width, target_height):
        """
        Calculate dimensions to maintain aspect ratio.

        Computes new dimensions and padding values to fit source image
        into target dimensions while preserving aspect ratio.

        Args:
            original_width (int): Original image width
            original_height (int): Original image height
            target_width (int): Target display width
            target_height (int): Target display height

        Returns:
            tuple: (new_width, new_height, pad_left, pad_top, pad_right, pad_bottom)
        """
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        if original_aspect > target_aspect:
            new_width = target_width
            new_height = int(target_width / original_aspect)
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top
            return new_width, new_height, 0, pad_top, 0, pad_bottom
        else:
            new_height = target_height
            new_width = int(target_height * original_aspect)
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            return new_width, new_height, pad_left, 0, pad_right, 0
    
    def draw_reid_results(self, frame, reid_results, camera_id):
        """
        Draw person re-identification results on frame with indicators and predictions.

        Renders bounding boxes, person names, confidence scores, cache status,
        exercise predictions, and various system information on the video frame.

        Args:
            frame (numpy.ndarray): Input video frame
            reid_results (list): List of person re-identification results
            camera_id (str): Camera identifier

        Returns:
            numpy.ndarray: Annotated frame with all visual information
        """
        camera = self.camera_manager.get_camera(camera_id)
        config = camera.get_config() if camera else {'name': f'Camera {camera_id}'}

        new_width, new_height, pad_left, pad_top, pad_right, pad_bottom = \
            self.calculate_display_dimensions(
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

        # Draw person re-identification results (using rtsp_reid_inference.py logic)
        for result in reid_results:
            bbox = result['bbox']
            track_id = result.get('track_id', 'N/A')
            person_name = result.get('person_name', 'Unknown')
            distance = result.get('distance', 999.0)
            is_cached = result.get('cached', False)
            exercise_info = result.get('exercise', None)  # Get exercise prediction

            # Scale and offset coordinates
            x1 = int(bbox[0] * scale_x) + pad_left
            y1 = int(bbox[1] * scale_y) + pad_top
            x2 = int(bbox[2] * scale_x) + pad_left
            y2 = int(bbox[3] * scale_y) + pad_top

            # Determine if recognized (same as rtsp_reid_inference.py)
            is_recognized = distance < self.matching_threshold

            # Color logic from rtsp_reid_inference.py:
            # Blue (cached), Green (match), Red (unknown)
            if is_cached:
                color = (255, 0, 0)  # Blue
                label = f"T{track_id}|ID{person_name}* d={distance:.2f}"
            elif is_recognized:
                color = (0, 255, 0)  # Green
                label = f"T{track_id}|ID{person_name} d={distance:.2f}"
            else:
                color = (0, 0, 255)  # Red
                label = f"T{track_id}|UNK d={distance:.2f}"

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw main label (same as rtsp_reid_inference.py)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw exercise prediction below the main label if available
            if exercise_info:
                exercise_text = f"Exercise: {exercise_info}"
                exercise_size = cv2.getTextSize(exercise_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Position below the main label
                exercise_y = y2 + label_size[1] + 20

                # Use orange background for exercise predictions
                exercise_bg_color = (0, 165, 255)  # Orange
                cv2.rectangle(display_frame, (x1, exercise_y),
                        (x1 + exercise_size[0] + 10, exercise_y + exercise_size[1] + 10),
                            exercise_bg_color, -1)
                cv2.putText(display_frame, exercise_text, (x1 + 5, exercise_y + exercise_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add a small exercise indicator icon (circle with "E")
                cv2.circle(display_frame, (x2 - 25, y1 + 25), 12, exercise_bg_color, -1)
                cv2.putText(display_frame, "E", (x2 - 30, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Source indicator circle (cache vs database)
            source_circle_color = (0, 255, 255) if is_cached else (255, 165, 0)  # Cyan for cache, Orange for DB
            cv2.circle(display_frame, (x2 - 10, y1 + 10), 5, source_circle_color, -1)

        # Add camera info
        camera_name = config.get('name', f'Camera {camera_id}')
        cv2.putText(display_frame, camera_name, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

        # Add FPS
        fps_text = f"FPS: {self.current_fps[camera_id]:.1f}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Add tracking and cache info
        active_tracks = len([r for r in reid_results if r.get('track_id') is not None])
        cached_tracks = len([r for r in reid_results if r.get('cached', False)])
        db_tracks = active_tracks - cached_tracks
        exercise_tracks = len([r for r in reid_results if r.get('exercise') is not None])

        track_info = f"Tracks: {active_tracks} (DB: {db_tracks}, Cache: {cached_tracks}, Exercise: {exercise_tracks})"
        cv2.putText(display_frame, track_info, (10, display_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Cache size info
        cache_size = len(self.cached_tracks)
        cache_info = f"Cached Tracks: {cache_size}"
        cv2.putText(display_frame, cache_info, (10, display_frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return display_frame

    def calculate_fps(self, camera_id):
        """
        Calculate frames per second for a specific camera.

        Updates FPS counter and calculates current FPS every second.

        Args:
            camera_id (str): Camera identifier
        """
        self.fps_counters[camera_id] += 1
        if time.time() - self.fps_start_times[camera_id] >= 1.0:
            self.current_fps[camera_id] = self.fps_counters[camera_id]
            self.fps_counters[camera_id] = 0
            self.fps_start_times[camera_id] = time.time()
    
    def update_stable_display(self):
        """
        Update stable display buffer for all cameras.

        Synchronizes processed frames from all cameras into a stable
        display buffer for consistent grid display rendering.
        """
        with self.display_lock:
            updated = False
            
            # Update all cameras
            for camera_id in self.camera_manager.get_camera_ids():
                camera = self.camera_manager.get_camera(camera_id)
                if camera:
                    processed_frame = camera.get_latest_processed_frame()
                    if processed_frame is not None:
                        self.stable_display_buffer[camera_id] = processed_frame
                        updated = True
                        self.calculate_fps(camera_id)
            
            if updated:
                self.stable_display_buffer['timestamp'] = time.time()
    
    def create_stable_display(self):
        """
        Create a grid display for all cameras.

        Combines individual camera frames into a single grid layout
        for multi-camera monitoring.

        Returns:
            numpy.ndarray: Combined grid display of all camera feeds
        """
        canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        with self.display_lock:
            camera_ids = self.camera_manager.get_camera_ids()
            
            for i, camera_id in enumerate(camera_ids):
                # Calculate grid position
                row = i // self.grid_cols
                col = i % self.grid_cols
                
                # Calculate pixel positions
                y_start = row * self.camera_height
                y_end = y_start + self.camera_height
                x_start = col * self.camera_width
                x_end = x_start + self.camera_width
                
                if self.stable_display_buffer.get(camera_id) is not None:
                    canvas[y_start:y_end, x_start:x_end] = self.stable_display_buffer[camera_id]
                else:
                    placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                    camera = self.camera_manager.get_camera(camera_id)
                    camera_name = camera.get_name() if camera else f"Camera {camera_id}"
                    cv2.putText(placeholder, f"Connecting {camera_name}...", (50, self.camera_height // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                    canvas[y_start:y_end, x_start:x_end] = placeholder
        return canvas
    
    def start_person_reid_system(self):
        """
        Start the person re-identification system.

        Initializes all cameras, starts processing threads, and begins
        the main display loop for the person re-identification system.

        Returns:
            bool: False if no cameras started successfully, otherwise runs until interrupted
        """
        print(f"Starting Person Re-Identification System ({self.num_cameras} cameras)...")
        
        # Start all cameras
        started_cameras = self.camera_manager.start_all_cameras()
        if started_cameras == 0:
            print("No cameras started successfully!")
            return False
        
        time.sleep(3)  # Wait for initialization
        self.running = True
        
        # Start processing threads for all cameras
        self.processing_threads = []
        for camera_id in self.camera_manager.get_camera_ids():
            thread = threading.Thread(target=self.processing_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.processing_threads.append((camera_id, thread))
        
        try:
            while self.running:
                # Update display buffer
                self.update_stable_display()
                
                # Create and show display
                display = self.create_stable_display()
                if not os.getenv('HEADLESS'):
                    cv2.imshow(f'Person Re-ID System ({self.num_cameras} cameras)', display)

                # Handle controls
                key = cv2.waitKey(33) & 0xFF
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_system()

    
    def stop_system(self):
        """
        Stop the person re-identification system.

        Gracefully shuts down all cameras, processing threads, and cleans up resources.
        """
        print("Stopping person re-identification system...")
        self.running = False
        
        # Stop all cameras
        self.camera_manager.stop_all_cameras()
        
        # Wait for processing threads to finish
        if hasattr(self, 'processing_threads'):
            for camera_id, thread in self.processing_threads:
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()