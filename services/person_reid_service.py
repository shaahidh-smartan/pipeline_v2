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
                 voting_threshold=4.5, voting_window=20, min_votes=10,
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

        self.logged_tracks = set()

        # Initialize utility modules
        self.db_manager = DatabaseManager()

        self.person_detector = PersonDetector()

        # Initialize BPBreID model
        self._initialize_reid(config_path, weights_path, gallery_dir, device)

        # Voting-based cache structures (replacing old cache)
        self.track_voting = {}  # {(camera_id, track_id): {'votes': {}, 'frame_count': 0, 'distances': {}}}
        self.cached_tracks = {}  # {(camera_id, track_id): {'person_id': str, 'distance': float, 'votes': int}}

        # Database embedding polling (disabled - using Redis streaming instead)
        self.last_embedding_check = 0
        self.embedding_check_interval = 5  # Check every 5 seconds

        # Note: Embeddings are now loaded via Redis streaming (see main1.py)
        # No need to poll database anymore

        # Create tracker arguments
        tracker_args = type('Args', (object,), {
            'track_thresh': 0.5,
            'match_thresh': 0.5,
            'buffer_size': 480,
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
        self.camera_width = 480
        self.camera_height = 480
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
        if gallery_dir and Path(gallery_dir).exists():
            gallery_path = Path(gallery_dir)
            embeddings_file = gallery_path / 'gallery_embeddings.pt'

            if embeddings_file.exists():
                self.gallery_embeddings = torch.load(embeddings_file, map_location=self.device)
                self.gallery_visibility = torch.load(gallery_path / 'gallery_visibility.pt', map_location=self.device)
                self.gallery_pids = torch.load(gallery_path / 'gallery_pids.pt', map_location=self.device)
            else:
                self._initialize_empty_gallery()
        else:
            self._initialize_empty_gallery()

    def _initialize_empty_gallery(self):
        """Initialize empty gallery tensors."""
        # Create empty tensors with proper shape: [0, num_parts, embedding_dim]
        # Assuming 6 body parts and 512-dim embeddings (BPBreID default)
        self.gallery_embeddings = torch.empty(0, 6, 512, device=self.device)
        self.gallery_visibility = torch.empty(0, 6, device=self.device)
        self.gallery_pids = torch.empty(0, dtype=torch.long, device=self.device)
        self.pid_to_user_id = {}  # Maps numeric PID -> firebase user ID string


    def initialize_database_connection(self):
        """
        Initialize and verify database connection.

        Establishes connection to the database and retrieves initial statistics
        for person embeddings and stored people.

        Raises:
            Exception: If database initialization fails
        """
        try:
            stats = self.db_manager.get_database_stats()

            self.db_stats['total_people'] = stats['person_people']
            self.db_stats['total_embeddings'] = stats['person_embeddings']
            self.db_stats['last_updated'] = time.time()


        except Exception as e:
            raise


    def load_person_embeddings(self, pt_path):
        """Load person embeddings from .pt files."""
        try:
            pt_dir = Path(pt_path)
            embeddings_file = pt_dir / 'embeddings.pt'
            visibility_file = pt_dir / 'visibility.pt'
            pids_file = pt_dir / 'pids.pt'

            if not (embeddings_file.exists() and visibility_file.exists() and pids_file.exists()):
                return None, None, None

            embeddings = torch.load(embeddings_file, map_location=self.device)
            visibility = torch.load(visibility_file, map_location=self.device)
            pids = torch.load(pids_file, map_location=self.device)


            return embeddings, visibility, pids

        except Exception as e:
            return None, None, None


    def add_to_gallery(self, embeddings, visibility, pids):
        """Add new embeddings to the gallery dynamically."""
        try:
            # Append to existing gallery
            self.gallery_embeddings = torch.cat([self.gallery_embeddings, embeddings], dim=0)
            self.gallery_visibility = torch.cat([self.gallery_visibility, visibility], dim=0)
            self.gallery_pids = torch.cat([self.gallery_pids, pids], dim=0)


            return True

        except Exception as e:
            return False

    def poll_and_load_new_embeddings(self):
        """
        Poll database for new embeddings and load them into gallery.

        NOTE: This method is deprecated. Embeddings are now loaded via Redis streaming.
        See main1.py's override and _embedding_consumer_loop for the new implementation.
        """
        pass  # No-op: Embeddings loaded via Redis streaming instead

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

    # --- Keypoint-based person signature for cache validation ---
    # COCO keypoint indices
    KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
    KP_L_ELBOW, KP_R_ELBOW = 7, 8
    KP_L_WRIST, KP_R_WRIST = 9, 10
    KP_L_HIP, KP_R_HIP = 11, 12
    KP_L_KNEE, KP_R_KNEE = 13, 14
    KP_L_ANKLE, KP_R_ANKLE = 15, 16

    # Bone pairs for computing normalized body proportions
    BONE_PAIRS = [
        (5, 6),    # shoulder width
        (11, 12),  # hip width
        (5, 7),    # left upper arm
        (6, 8),    # right upper arm
        (7, 9),    # left forearm
        (8, 10),   # right forearm
        (11, 13),  # left upper leg
        (12, 14),  # right upper leg
        (13, 15),  # left lower leg
        (14, 16),  # right lower leg
    ]

    @staticmethod
    def compute_keypoint_signature(keypoints, min_kp_conf=0.3):
        """
        Compute a normalized bone-length signature from COCO 17 keypoints.
        This signature represents body proportions and is unique per person.

        Args:
            keypoints: numpy array [17, 3] with (x, y, confidence)
            min_kp_conf: minimum keypoint confidence to use

        Returns:
            numpy array of normalized bone lengths, or None if insufficient keypoints
        """
        if keypoints is None or keypoints.shape[0] < 17:
            return None

        bone_lengths = []
        valid_count = 0
        for (a, b) in PersonReIDService.BONE_PAIRS:
            if keypoints[a, 2] >= min_kp_conf and keypoints[b, 2] >= min_kp_conf:
                dx = keypoints[a, 0] - keypoints[b, 0]
                dy = keypoints[a, 1] - keypoints[b, 1]
                bone_lengths.append(np.sqrt(dx * dx + dy * dy))
                valid_count += 1
            else:
                bone_lengths.append(0.0)

        # Need at least 4 valid bones for a meaningful signature
        if valid_count < 4:
            return None

        bone_lengths = np.array(bone_lengths, dtype=np.float32)

        # Normalize by the sum of valid bones (scale-invariant)
        total = bone_lengths[bone_lengths > 0].sum()
        if total < 1e-6:
            return None
        bone_lengths = bone_lengths / total

        return bone_lengths

    @staticmethod
    def compare_keypoint_signatures(sig1, sig2):
        """
        Compare two keypoint signatures. Returns similarity 0-1.
        Only compares bones where both signatures have valid values (>0).
        """
        if sig1 is None or sig2 is None:
            return 0.0

        # Only compare where both have valid bones
        valid = (sig1 > 0) & (sig2 > 0)
        if valid.sum() < 3:
            return 0.0

        diff = np.abs(sig1[valid] - sig2[valid])
        similarity = 1.0 - np.mean(diff) * 5.0  # Scale so typical variance maps to ~0.7-0.9
        return max(0.0, min(1.0, similarity))

    def match_detection_to_track(self, person_detections, track_bbox):
        """
        Find the detection that best matches a tracked bbox by IoU.
        Returns the keypoints from the best matching detection, or None.
        """
        tx1, ty1, tx2, ty2 = track_bbox
        best_iou = 0.0
        best_kps = None

        for det in person_detections:
            dx1, dy1, dx2, dy2 = det['bbox']
            # Compute IoU
            ix1 = max(tx1, dx1); iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2); iy2 = min(ty2, dy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_t = max(1, (tx2 - tx1) * (ty2 - ty1))
            area_d = max(1, (dx2 - dx1) * (dy2 - dy1))
            iou = inter / (area_t + area_d - inter)
            if iou > best_iou:
                best_iou = iou
                best_kps = det.get('keypoints')

        return best_kps if best_iou > 0.3 else None


    def cache_master_list(self, person_name, camera_id, track_id):
        """
        Append cached person to master list and save to database once per unique track.

        Saves person identification to both master list and database, ensuring
        each unique track is only logged once. The person_name is actually the
        PID (person_id) from the BPBreID gallery.

        Args:
            person_name (str): String representation of person_id (PID) from gallery
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

                # Insert into database (global_id is integer column, pass None if no numeric ID)
                try:
                    db_success = self.db_manager.insert_person_reid_log(
                        person_name, camera_id, track_id, global_id=None
                    )

                    if db_success:
                        pass
                    else:
                        pass

                    # Mark this track as saved
                    self.logged_tracks.add(track_key)

                except Exception as e:
                    pass

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

                        # Only clear voting (not cached tracks) on empty frame
                        keys_to_remove = [key for key in self.track_voting.keys() if key[0] == camera_id]
                        for key in keys_to_remove:
                            del self.track_voting[key]

                    except Exception as e:
                        pass

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

                # Get current keypoints for this track
                current_kps = self.match_detection_to_track(person_detections, [x1, y1, x2, y2])
                current_sig = self.compute_keypoint_signature(current_kps)

                # Check if this track is already cached
                if cache_key in self.cached_tracks:
                    # Keypoint-based cache validation to detect track ID swaps
                    cached_sig = self.cached_tracks[cache_key].get('kp_signature')
                    cache_valid = True

                    if current_sig is not None and cached_sig is not None:
                        sim = self.compare_keypoint_signatures(current_sig, cached_sig)

                        if sim < 0.5:
                            # Signature mismatch — possible track ID swap
                            # Check if another cached track matches this person better
                            best_swap_key = None
                            best_swap_sim = sim  # current (bad) similarity

                            for other_key, other_data in self.cached_tracks.items():
                                if other_key == cache_key or other_key[0] != cam_id:
                                    continue
                                other_sig = other_data.get('kp_signature')
                                if other_sig is not None:
                                    other_sim = self.compare_keypoint_signatures(current_sig, other_sig)
                                    if other_sim > best_swap_sim and other_sim > 0.6:
                                        best_swap_sim = other_sim
                                        best_swap_key = other_key

                            if best_swap_key is not None:
                                # Swap the two caches to correct the track ID mix-up
                                self.cached_tracks[cache_key], self.cached_tracks[best_swap_key] = \
                                    self.cached_tracks[best_swap_key], self.cached_tracks[cache_key]
                            else:
                                # No swap partner found — invalidate this cache, re-vote
                                del self.cached_tracks[cache_key]
                                if cache_key in self.track_voting:
                                    del self.track_voting[cache_key]
                                cache_valid = False

                    if cache_valid and cache_key in self.cached_tracks:
                        person_name = self.cached_tracks[cache_key]['person_id']
                        result_distance = self.cached_tracks[cache_key]['distance']
                        self.cached_tracks[cache_key]['bbox'] = [x1, y1, x2, y2]
                        # Update keypoint signature with exponential moving average
                        if current_sig is not None:
                            if cached_sig is not None:
                                self.cached_tracks[cache_key]['kp_signature'] = 0.8 * cached_sig + 0.2 * current_sig
                            else:
                                self.cached_tracks[cache_key]['kp_signature'] = current_sig

                        # Re-collect embedding when pose/aspect ratio changes significantly
                        cur_w = max(1, x2 - x1)
                        cur_h = max(1, y2 - y1)
                        cur_ar = cur_w / cur_h
                        cached_ar = self.cached_tracks[cache_key].get('initial_ar')
                        last_recollect = self.cached_tracks[cache_key].get('last_recollect', 0)

                        if cached_ar is not None:
                            ar_change = abs(cur_ar - cached_ar) / max(cached_ar, 0.01)
                            now_rc = time.time()
                            # Re-collect if aspect ratio changed >40% and at least 5s since last
                            if ar_change > 0.4 and (now_rc - last_recollect) > 5.0:
                                try:
                                    rc_crop = self.person_detector.get_person_crop(frame, [x1, y1, x2, y2], padding=10)
                                    if rc_crop is not None and self.gallery_embeddings.shape[0] > 0:
                                        rc_rgb = cv2.cvtColor(rc_crop, cv2.COLOR_BGR2RGB)
                                        rc_output = self.extractor([rc_rgb])
                                        rc_emb, rc_vis = self.extract_test_embeddings(rc_output)
                                        rc_emb = rc_emb.to(self.device)
                                        rc_vis = rc_vis.to(self.device)

                                        # Find the PID for this person_name
                                        rc_pid = None
                                        for pid, name in self.pid_to_user_id.items():
                                            if name == person_name:
                                                rc_pid = pid
                                                break

                                        if rc_pid is not None:
                                            rc_pids = torch.tensor([rc_pid], dtype=torch.long, device=self.device)
                                            self.add_to_gallery(rc_emb, rc_vis, rc_pids)
                                            self.cached_tracks[cache_key]['last_recollect'] = now_rc
                                            self.cached_tracks[cache_key]['initial_ar'] = cur_ar
                                            print(f"[RECOLLECT] {cam_id}:T{track_id} '{person_name}' AR {cached_ar:.2f}->{cur_ar:.2f} (change {ar_change:.0%}) | gallery size: {self.gallery_embeddings.shape[0]}")
                                except Exception as e:
                                    print(f"[RECOLLECT_ERR] {cam_id}:T{track_id} - {e}")
                        else:
                            self.cached_tracks[cache_key]['initial_ar'] = cur_ar

                        is_cached = True
                    else:
                        # Cache was invalidated — fall through to re-identification below
                        cache_valid = False

                if cache_key not in self.cached_tracks:
                    # Try to inherit cache from a recently disappeared track with overlapping bbox
                    inherited = False
                    current_bbox = [x1, y1, x2, y2]
                    for old_key, old_data in list(self.cached_tracks.items()):
                        if old_key[0] == cam_id and 'last_seen' in old_data and 'bbox' in old_data:
                            # Check spatial overlap with disappeared track's last bbox
                            ox1, oy1, ox2, oy2 = old_data['bbox']
                            ix1 = max(x1, ox1); iy1 = max(y1, oy1)
                            ix2 = min(x2, ox2); iy2 = min(y2, oy2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                area_new = max(1, (x2 - x1) * (y2 - y1))
                                if inter / area_new > 0.3:
                                    # Transfer cache to new track (preserve keypoint signature)
                                    self.cached_tracks[cache_key] = {
                                        'person_id': old_data['person_id'],
                                        'distance': old_data['distance'],
                                        'votes': old_data['votes'],
                                        'kp_signature': old_data.get('kp_signature')
                                    }
                                    del self.cached_tracks[old_key]
                                    person_name = old_data['person_id']
                                    result_distance = old_data['distance']
                                    is_cached = True
                                    inherited = True
                                    break

                    if inherited:
                        pass  # Already set above
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

                            person_name = self.pid_to_user_id.get(best_pid, str(best_pid))
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
                                        # Prevent duplicate: skip if another active track on same camera already has this identity
                                        already_taken = False
                                        for ck, cd in self.cached_tracks.items():
                                            if ck[0] == cam_id and ck != cache_key and 'last_seen' not in cd:
                                                if cd.get('person_id') == max_votes_id:
                                                    already_taken = True
                                                    break

                                        if not already_taken:
                                            init_w = max(1, x2 - x1)
                                            init_h = max(1, y2 - y1)
                                            self.cached_tracks[cache_key] = {
                                                'person_id': max_votes_id,
                                                'distance': voting_entry['distances'][max_votes_id],
                                                'votes': max_votes,
                                                'bbox': [x1, y1, x2, y2],
                                                'kp_signature': current_sig,
                                                'initial_ar': init_w / init_h,
                                                'last_recollect': 0
                                            }

                                            person_name = max_votes_id
                                            result_distance = voting_entry['distances'][max_votes_id]
                                            is_cached = True

                                            # Save to database (person_reid_mapped table)
                                            self.cache_master_list(max_votes_id, cam_id, track_id)

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

            # Deduplicate: only one track per camera can claim a given identity
            # The track with the lowest distance wins; others become UNKNOWN
            name_best = {}  # {person_name: index of best result}
            for i, r in enumerate(reid_results):
                name = r['person_name']
                if name == "UNKNOWN":
                    continue
                if name not in name_best or r['distance'] < reid_results[name_best[name]]['distance']:
                    name_best[name] = i

            for i, r in enumerate(reid_results):
                name = r['person_name']
                if name != "UNKNOWN" and name_best.get(name) != i:
                    reid_results[i]['person_name'] = "UNKNOWN"
                    reid_results[i]['match'] = None
                    # Also remove the duplicate from cache
                    dup_key = (r['camera_id'], r['track_id'])
                    if dup_key in self.cached_tracks:
                        del self.cached_tracks[dup_key]

            # Cleanup disappeared tracks with time-based retention
            active_track_ids = {(camera_id, target[0]) for target in target_box}
            now_cleanup = time.time()

            # Mark disappearance time instead of instant deletion
            for key in list(self.cached_tracks.keys()):
                if key[0] == camera_id and key not in active_track_ids:
                    if 'last_seen' not in self.cached_tracks[key]:
                        self.cached_tracks[key]['last_seen'] = now_cleanup
                    elif now_cleanup - self.cached_tracks[key]['last_seen'] > 20.0:
                        del self.cached_tracks[key]
                elif key in active_track_ids:
                    # Track is active again, refresh last_seen
                    self.cached_tracks[key].pop('last_seen', None)

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
            pass

    def processing_thread(self, camera_id):
        """
        Main processing thread for each camera.

        Runs continuously to process frames from a specific camera,
        perform person detection and re-identification, and update display.

        Args:
            camera_id (str): Camera identifier
        """
        
        # Initialize persistent detection data for this camera
        persistent_detections = []
        detection_timestamp = 0
        
        while self.running:
            try:
                camera = self.camera_manager.get_camera(camera_id)
                if not camera:
                    time.sleep(0.1)
                    continue

                record = camera.get_frame()

                if record is not None:
                    # Extract frame from record dict (camera.py now returns dict with metadata)
                    frame = record["frame"] if isinstance(record, dict) else record
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

            # Draw exercise prediction below the bounding box if available
            if exercise_info:
                exercise_text = f"Exercise: {exercise_info}"
                (exercise_w, exercise_h), _ = cv2.getTextSize(exercise_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Position below the bounding box
                exercise_y = y2 + 5

                # Use orange background for exercise predictions
                exercise_bg_color = (0, 165, 255)  # Orange
                cv2.rectangle(display_frame, (x1, exercise_y),
                        (x1 + exercise_w + 10, exercise_y + exercise_h + 10),
                            exercise_bg_color, -1)
                cv2.putText(display_frame, exercise_text, (x1 + 5, exercise_y + exercise_h + 5),
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
        
        # Start all cameras
        started_cameras = self.camera_manager.start_all_cameras()
        if started_cameras == 0:
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
            pass
        finally:
            self.stop_system()

    
    def stop_system(self):
        """
        Stop the person re-identification system.

        Gracefully shuts down all cameras, processing threads, and cleans up resources.
        """
        self.running = False
        
        # Stop all cameras
        self.camera_manager.stop_all_cameras()
        
        # Wait for processing threads to finish
        if hasattr(self, 'processing_threads'):
            for camera_id, thread in self.processing_threads:
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()