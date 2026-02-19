#!/usr/bin/env python3
"""
main.py with database integration for pose, exercise, and weight data
"""
import os
import sys

# Fix libproxy crash with RTSP URLs containing special characters
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

import csv
import signal
import time
from typing import List, Dict
import cv2
cv2.setNumThreads(0)  # Fix segfault with GStreamer
import numpy as np
from datetime import datetime
from collections import defaultdict
import json
from services.enhanced_mvit_engine import MViTEngine
from services.person_reid_service import PersonReIDService
from services.RepNet_engine import RepNetEngine
from camera import create_camera_configs_from_ips
from utils.database_manager import DatabaseManager
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import base64
import asyncio
import websockets
import pickle

database_checked = False

# ==========================================
# SMPL Communication Configuration
# ==========================================
# Redis configuration (PRIMARY - recommended for performance)
USE_REDIS = True  # Set to False to use WebSocket instead
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_STREAM = "smpl_frames"
REDIS_MAX_QUEUE_LEN = 100  # Prevent queue overflow

# ==========================================
# Person Embedding Redis Configuration
# ==========================================
REDIS_EMBEDDING_STREAM = "person_embeddings"  # Stream for receiving embeddings from collector

# Websocket configuration (FALLBACK - kept for compatibility)
WS_URL = "ws://localhost:8765"

VISUAL_MODE = True # Set to False to test if OpenCV drawing/display causes bottleneck

def clamp_bbox(bbox, h, w):
    """
    Clamp bounding box coordinates to ensure they are within image boundaries.

    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        h (int): Image height
        w (int): Image width

    Returns:
        tuple or None: Clamped bounding box coordinates (x1, y1, x2, y2) or None if invalid
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w,     int(x2)))
    y2 = max(0, min(h,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

@dataclass
class ExerciseEntry:
    """Single exercise entry with frames and metadata"""
    exercise: str
    confidence: float
    frames: List 
    timestamp: float
    voting_cycle_id: str
    vote_counts: dict
    batches_used: int
    batch_ids: list
    weight: str = "unknown"
    weight_confidence: float = 0.0
    processed_by_repnet: bool = False
    reps: Optional[int] = None
    rep_conf: Optional[float] = None
    repnet_stride: Optional[int] = None

class GlobalExerciseTracker:
    """Global tracker that maintains a master list of all exercises per track id."""
    
    def __init__(self):
        self._master_exercise_list: Dict[Tuple[str, int], List[ExerciseEntry]] = defaultdict(list)
        self._repnet_sent_index: Dict[Tuple[str, int], int] = defaultdict(int)
        self._lock = threading.Lock()
        self.total_exercises_added = 0
        self.total_repnet_processed = 0

    def add_exercise_entry(self, cam_id: str, track_id: int, exercise_entry: ExerciseEntry) -> None:
        """
        Add a new exercise entry to the master list for a specific track id.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier
            exercise_entry (ExerciseEntry): Exercise entry data to add
        """
        key = (cam_id, track_id)
        
        with self._lock:
            self._master_exercise_list[key].append(exercise_entry)
            self.total_exercises_added += 1
            entry_index = len(self._master_exercise_list[key]) - 1
            
    def get_entry_for_repnet(self, cam_id: str, track_id: int) -> Optional[Tuple[ExerciseEntry, int]]:
        """
        Get the latest unprocessed exercise entry and its index for RepNet processing.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier

        Returns:
            Optional[Tuple[ExerciseEntry, int]]: Tuple of (exercise_entry, index) or None if no unprocessed entries
        """
        key = (cam_id, track_id)
        
        with self._lock:
            exercise_list = self._master_exercise_list.get(key, [])
            last_sent_index = self._repnet_sent_index.get(key, -1)
            
            for i in range(last_sent_index + 1, len(exercise_list)):
                entry = exercise_list[i]
                if not entry.processed_by_repnet:
                    self._repnet_sent_index[key] = i
                    return entry, i
            return None

    def mark_repnet_processed(self, cam_id: str, track_id: int, entry_index: int,
                        reps: int, rep_conf: float, stride: int) -> None:
        """
        Update the SAME entry in master dict with RepNet results.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier
            entry_index (int): Index of the exercise entry to update
            reps (int): Number of repetitions detected
            rep_conf (float): Confidence score for repetition count
            stride (int): Stride used in RepNet processing
        """
        key = (cam_id, track_id)
        
        with self._lock:
            exercise_list = self._master_exercise_list.get(key, [])
            
            if 0 <= entry_index < len(exercise_list):
                entry = exercise_list[entry_index]
                entry.processed_by_repnet = True
                entry.reps = reps
                entry.rep_conf = rep_conf
                entry.repnet_stride = stride
                self.total_repnet_processed += 1

    def get_track_history(self, cam_id: str, track_id: int) -> List[ExerciseEntry]:
        """
        Get complete exercise history for a track.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier

        Returns:
            List[ExerciseEntry]: Copy of all exercise entries for the track
        """
        key = (cam_id, track_id)
        with self._lock:
            return self._master_exercise_list.get(key, []).copy()

    def get_track_summary(self, cam_id: str, track_id: int) -> dict:
        """
        Get summary statistics for a track.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier

        Returns:
            dict: Dictionary containing track statistics including total exercises,
                  processed count, pending count, and exercise list
        """
        key = (cam_id, track_id)
        
        with self._lock:
            exercise_list = self._master_exercise_list.get(key, [])
            last_sent_index = self._repnet_sent_index.get(key, -1)
            processed_count = sum(1 for entry in exercise_list if entry.processed_by_repnet)
            pending_count = len(exercise_list) - last_sent_index - 1
            
            return {
                'total_exercises': len(exercise_list),
                'processed_by_repnet': processed_count,
                'last_sent_index': last_sent_index,
                'pending_for_repnet': pending_count,
                'exercises': [entry.exercise for entry in exercise_list]
            }

    def get_global_stats(self) -> dict:
        """
        Get global statistics across all tracks.

        Returns:
            dict: Dictionary containing global statistics including active tracks count,
                  total exercises, and per-track exercise counts
        """
        with self._lock:
            active_tracks = len(self._master_exercise_list)
            total_exercises = sum(len(exercises) for exercises in self._master_exercise_list.values())
            
            return {
                'active_tracks': active_tracks,
                'total_exercises_in_master_list': total_exercises,
                'total_exercises_added': self.total_exercises_added,
                'total_repnet_processed': self.total_repnet_processed,
                'tracks': {f"{k[0]}:{k[1]}": len(v) for k, v in self._master_exercise_list.items()}
            }

# Global instance
global_exercise_tracker = GlobalExerciseTracker()

class BridgeReIDService(PersonReIDService):
    """BridgeReIDService with database integration for pose, exercise, and weight data"""

    def __init__(self, stream_configs, config_path, weights_path, gallery_dir=None,
                 voting_threshold=4.5, voting_window=30, min_votes=15,
                 matching_threshold=6.0, device='cuda',
                 mvit_engine=None, repnet_engine=None, visual_mode=True):
        # Initialize BPBReID base class
        super().__init__(
            stream_configs=stream_configs,
            config_path=config_path,
            weights_path=weights_path,
            gallery_dir=gallery_dir,
            voting_threshold=voting_threshold,
            voting_window=voting_window,
            min_votes=min_votes,
            matching_threshold=matching_threshold,
            device=device
        )

        self.visual_mode = visual_mode

        self.sf = mvit_engine  # Keep as self.sf for compatibility
        self.repnet = repnet_engine
        self.enable_repnet = self.repnet is not None
        self.enable_slowfast = self.sf is not None  # Name stays for compatibility
        
        self.exercise_tracker = global_exercise_tracker
        self._rep_overlay = {}  # (cam,tid) -> latest RepNet result for display
        self.ex_conf_threshold = 0.4

        # Note: self.db_manager is already initialized by parent PersonReIDService

        # CSV files (backup logging)
        self.csv_path = "exercise_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","camera_id","track_id","exercise","exercise_conf",
                             "reps","rep_conf","frame_count","voting_cycle_id",
                             "vote_counts","batches_used","batch_ids","global_counters","entry_index", "weight","weight_conf"])
        
        self.weight_csvpath = "weight_detections.csv"
        with open(self.weight_csvpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "camera_id", "track_id", "weight_label", "confidence", "bbox"])

        # Weight detection
        self.weight_engine = None
        self.weight_detection_stride = 1
        self.frame_counters = {}
        self.last_weight_detections = {}
        self.init_weight_engine()

        # SMPL Communication setup
        # Per-track frame counter that mirrors MViT's global_counter logic
        # This tracks how many 64-frame windows have been created per track
        self.smpl_track_frame_counter = {}  # (cam_id, track_id) -> frame count within current window
        self.smpl_track_window_counter = {}  # (cam_id, track_id) -> global window counter (matches MViT)

        # Full-frame buffer indexed by global_counter for SMPL
        # Structure: {(cam_id, track_id): {global_counter: [(frame, reid_result), ...]}}
        self.smpl_global_frame_buffer = defaultdict(lambda: defaultdict(list))
        self.smpl_sent_for_voting_cycle = set()  # Track which voting cycles we've already sent

        # Redis connection for SMPL (PRIMARY)
        self.redis_client = None
        if USE_REDIS:
            self._init_redis_connection()

        # Redis embedding consumer thread
        self.embedding_consumer_thread = None
        self.embedding_consumer_running = False
        if USE_REDIS and self.redis_client is not None:
            self._start_embedding_consumer()

        # Websocket connection for SMPL (FALLBACK - commented out by default)
        # Uncomment the block below to use WebSocket instead of Redis
        # self.ws_connection = None
        # self.ws_lock = threading.Lock()
        # self.ws_loop = None
        # self.ws_thread = None
        # self._start_websocket_loop()

    def init_weight_engine(self):
        """
        Initialize weight recognition engine if model files are available.

        Attempts to load weight recognition models and creates WeightRecognitionEngine instance.
        Falls back gracefully if models are not found or initialization fails.
        """
        try:
            required_files = ["models/singledb.pt", "models/new_embed.json"]
            if all(os.path.exists(f) for f in required_files):
                from services.weight_recognition_module import WeightRecognitionEngine
                self.weight_engine = WeightRecognitionEngine()
            else:
                self.weight_engine = None
        except Exception as e:
            self.weight_engine = None

    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list): First bounding box [x1, y1, x2, y2]
            box2 (list): Second bounding box [x1, y1, x2, y2]

        Returns:
            float: IoU value between 0.0 and 1.0
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def point_in_box(self, point, box):
        """
        Check if a point is inside a bounding box.

        Args:
            point (tuple): Point coordinates (x, y)
            box (list): Bounding box [x1, y1, x2, y2]

        Returns:
            bool: True if point is inside the box, False otherwise
        """
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def associate_weights_to_tracks(self, weight_detections, track_boxes):
        """
        Associate weight detections to person tracks based on spatial proximity.

        Args:
            weight_detections (list): List of weight detection dictionaries
            track_boxes (list): List of track bounding boxes [track_id, x, y, w, h]

        Returns:
            dict: Dictionary mapping track_id to list of associated weight detections
        """
        associations = defaultdict(list)
        
        for weight_det in weight_detections:
            weight_bbox = weight_det['bbox']
            weight_center = (
                (weight_bbox[0] + weight_bbox[2]) / 2,
                (weight_bbox[1] + weight_bbox[3]) / 2
            )
            
            best_track_id = None
            best_iou = 0.0
            
            for track_data in track_boxes:
                track_id = track_data[0]
                x, y, w, h = track_data[1:5]
                track_bbox = [x, y, x + w, y + h]
                
                center_inside = self.point_in_box(weight_center, track_bbox)
                iou = self.compute_iou(weight_bbox, track_bbox)
                
                if center_inside or iou >= 0.15:
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None and len(associations[best_track_id]) < 2:
                associations[best_track_id].append(weight_det)
        
        return dict(associations)

    def log_to_csv(self, entry: ExerciseEntry, cam_id: str, track_id: int, entry_index: int):
        """
        Log exercise entry to CSV file and database.

        Args:
            entry (ExerciseEntry): Exercise entry to log
            cam_id (str): Camera identifier
            track_id (int): Track identifier
            entry_index (int): Index of the entry in the track history
        """
        if not entry.processed_by_repnet:
            return
            
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # CSV logging (backup)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, cam_id, track_id, entry.exercise, f"{entry.confidence:.3f}",
                entry.reps or "", f"{entry.rep_conf:.3f}" if entry.rep_conf else "",
                len(entry.frames), entry.voting_cycle_id,
                str(entry.vote_counts), entry.batches_used, 
                str(entry.batch_ids),
                str(getattr(entry, 'global_counters', [])),
                entry_index,
                entry.weight if hasattr(entry, 'weight') else "unknown",
                f"{entry.weight_confidence:.3f}" if hasattr(entry, 'weight_confidence') else ""
            ])
        
        # Database logging (primary)
        try:
            exercise_data = {
                'timestamp': ts,
                'camera_id': cam_id,
                'track_id': track_id,
                'exercise': entry.exercise,
                'exercise_conf': entry.confidence,
                'reps': entry.reps,
                'rep_conf': entry.rep_conf,
                'frame_count': len(entry.frames),
                'voting_cycle_id': entry.voting_cycle_id,
                'vote_counts': entry.vote_counts,
                'batches_used': entry.batches_used,
                'batch_ids': entry.batch_ids,
                'global_counters': getattr(entry, 'global_counters', []),
                'entry_index': entry_index,
                'weight': entry.weight if hasattr(entry, 'weight') else 'unknown',
                'weight_conf': entry.weight_confidence if hasattr(entry, 'weight_confidence') else 0.0
            }
            
            success = self.db_manager.insert_exercise_log(exercise_data)
            if success:
                pass
                
        except Exception as db_error:
            pass

    def poll_and_load_new_embeddings(self):
        """
        Override parent method - no-op since we use Redis streaming instead of DB polling.

        Embeddings are now received via Redis stream consumer thread (_embedding_consumer_loop)
        which provides instant updates without polling overhead.
        """
        pass  # No-op: Redis consumer handles embedding loading

    def process_frame_for_reid(self, frame, camera_id):
        """
        Enhanced process_frame_for_reid with database integration.

        Processes a frame for person re-identification, pose analysis, weight detection,
        and exercise recognition with full database logging.

        Args:
            frame (numpy.ndarray): Input video frame
            camera_id (str): Camera identifier

        Returns:
            list: List of person detection results with tracking and exercise information
        """

        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0
        self.frame_counters[camera_id] += 1
        
        results = super().process_frame_for_reid(frame, camera_id)

        if not results or frame is None:
            return results

        # Submit crops to SlowFast
        H, W = frame.shape[:2]
        for r in results:
            try:
                bbox = r.get("bbox")
                if not bbox:
                    continue

                clamped = clamp_bbox(bbox, H, W)
                if not clamped:
                    continue

                x1, y1, x2, y2 = clamped
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr is None or crop_bgr.size == 0:
                    continue

                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

                cam = r.get("camera_id", camera_id)
                tid = int(r.get("track_id", -1))

                # Submit 224x224 RGB crop to SlowFast
                if tid >= 0 and self.enable_slowfast:
                    self.sf.submit_crop(cam, tid, rgb224)

                    # Buffer full frame for SMPL (synchronized with MViT's 64-frame windows)
                    # This stores the full frame indexed by global_counter
                    if USE_REDIS and self.redis_client is not None:
                        self.buffer_frame_for_smpl(frame, r, cam, tid)

            except Exception as e:
                continue

        # Weight detection with database integration
        if self.weight_engine is not None and self.frame_counters[camera_id] % self.weight_detection_stride == 0:
            try:
                weight_detections = self.weight_engine.detect_all_weights_frame(frame)
                self.last_weight_detections[camera_id] = weight_detections
                
                if weight_detections:
                    track_boxes = []
                    for result in results:
                        track_id = result.get('track_id')
                        bbox = result.get('bbox')
                        if track_id is not None and bbox is not None:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                            track_boxes.append([track_id, x1, y1, w, h])
                    
                    if track_boxes:
                        weight_assignments = self.associate_weights_to_tracks(weight_detections, track_boxes)
                        
                        # Log each detection
                        current_time = datetime.now().isoformat()
                        for track_id, assigned_weights in weight_assignments.items():
                            for weight_det in assigned_weights:
                                label = weight_det['label']
                                conf = weight_det['conf_emb']
                                bbox_str = f"{weight_det['bbox'][0]},{weight_det['bbox'][1]},{weight_det['bbox'][2]},{weight_det['bbox'][3]}"
                                
                                # Database insertion
                                weight_data = {
                                    'timestamp': current_time,
                                    'camera_id': camera_id,
                                    'track_id': track_id,
                                    'weight_label': label,
                                    'confidence': conf,
                                    'bbox': bbox_str
                                }
                                
                                weight_success = self.db_manager.insert_weight_detection(weight_data)
                                if weight_success:
                                    pass
                                
                                # CSV backup
                                with open(self.weight_csvpath, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([current_time, camera_id, track_id, label, f"{conf:.3f}", bbox_str])
                                
                                # Add to result for display
                                for result in results:
                                    if result.get('track_id') == track_id:
                                        result['weight_current'] = f"{label} ({conf:.3f})"
                                        break
                        
            except Exception as e:
                self.last_weight_detections[camera_id] = []
        else:
            weight_detections = self.last_weight_detections.get(camera_id, [])

        # Process SlowFast voting results
        if self.enable_slowfast and self.enable_repnet:
            self.process_voting_results()

        return results

    def _submit_crops_for_exercise(self, frame, reid_results, camera_id):
        """
        Override base class method to submit person crops for exercise detection.

        This method extracts person crops from the frame and submits them to the
        SlowFast exercise recognition engine.

        Args:
            frame: The video frame
            reid_results: List of ReID results with bbox and track_id
            camera_id: Camera identifier
        """
        if not self.enable_slowfast:
            return

        if not reid_results or frame is None:
            return

        # Submit crops to SlowFast (same logic as process_frame_for_reid)
        H, W = frame.shape[:2]
        for r in reid_results:
            try:
                bbox = r.get("bbox")
                if not bbox:
                    continue

                clamped = clamp_bbox(bbox, H, W)
                if not clamped:
                    continue

                x1, y1, x2, y2 = clamped
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr is None or crop_bgr.size == 0:
                    continue

                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

                cam = r.get("camera_id", camera_id)
                tid = int(r.get("track_id", -1))

                # Submit 224x224 RGB crop to SlowFast
                if tid >= 0:
                    self.sf.submit_crop(cam, tid, rgb224)

                    # Buffer full frame for SMPL (synchronized with MViT's 64-frame windows)
                    if USE_REDIS and self.redis_client is not None:
                        self.buffer_frame_for_smpl(frame, r, cam, tid)

            except Exception as e:
                continue

    def process_voting_results(self):
        """
        Process all voting results from SlowFast engine and add to global tracker.

        Retrieves pending voting results from SlowFast engine, creates ExerciseEntry objects,
        and adds them to the global exercise tracker for RepNet processing.
        """
        try:
            all_voting_results = self.sf.get_all_pending_voting_results()
            
            if all_voting_results:
                for (cam, tid), voting_result in all_voting_results.items():
                    exercise_entry = ExerciseEntry(
                        exercise=voting_result['exercise'],
                        confidence=voting_result['confidence'],
                        frames=voting_result['frames'],
                        timestamp=voting_result['timestamp'],
                        voting_cycle_id=f"{cam}_{tid}_{int(time.time())}",
                        vote_counts=voting_result['vote_counts'],
                        batches_used=voting_result['batches_used'],
                        batch_ids=voting_result['batch_ids'],
                        weight=voting_result.get('weight', 'unknown'),
                        weight_confidence=voting_result.get('weight_confidence', 0.0)
                    )
                    
                    # Add global_counters to the entry
                    exercise_entry.global_counters = voting_result.get('global_counters', [])

                    # IMMEDIATELY update overlay with voting result (before RepNet processing)
                    # This ensures display shows current exercise without waiting for RepNet
                    key = (cam, tid)
                    self._rep_overlay[key] = {
                        "exercise": voting_result['exercise'],
                        "exercise_conf": voting_result['confidence'],
                        "reps": None,  # Will be updated by RepNet later
                        "conf": 0.0,
                        "ts": time.time(),
                        "voting_cycle_id": exercise_entry.voting_cycle_id,
                        "entry_index": None  # Will be set after adding to tracker
                    }

                    self.exercise_tracker.add_exercise_entry(cam, tid, exercise_entry)

                    # Send sampled frames to SMPL using global_counters
                    # This samples every 5th frame from the 192 frames (3 x 64-frame windows)
                    global_counters = voting_result.get('global_counters', [])
                    if global_counters and USE_REDIS and self.redis_client is not None:
                        self.send_sampled_frames_to_smpl(
                            cam, tid, global_counters,
                            voting_result['exercise'],
                            exercise_entry.voting_cycle_id
                        )
            
            self.process_pending_repnet_entries()
                        
        except Exception as e:
            pass

    def process_pending_repnet_entries(self):
        """
        Process any entries that need RepNet processing.

        Iterates through all active tracks and processes exercise entries
        that have not yet been analyzed by RepNet.
        """
        global_stats = self.exercise_tracker.get_global_stats()
        
        for track_key in global_stats.get('tracks', {}):
            cam_id, track_id = track_key.split(':')
            track_id = int(track_id)
            
            entry_data = self.exercise_tracker.get_entry_for_repnet(cam_id, track_id)
            if entry_data:
                entry, entry_index = entry_data
                self.process_with_repnet(cam_id, track_id, entry, entry_index)

    def process_with_repnet(self, cam_id: str, track_id: int, exercise_entry: ExerciseEntry, entry_index: int):
        """
        Process exercise entry with RepNet to count repetitions.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier
            exercise_entry (ExerciseEntry): Exercise entry to process
            entry_index (int): Index of the entry in the track history
        """
        try:
            frames_112 = [cv2.resize(f, (112, 112), interpolation=cv2.INTER_AREA) 
                         for f in exercise_entry.frames]
            
            out = self.repnet.infer_clip(clip_rgb=frames_112)
            
            reps = out.get("reps")
            rep_conf = out.get("rep_conf", 0.0)
            stride = out.get("stride", 1)
            
            self.exercise_tracker.mark_repnet_processed(cam_id, track_id, entry_index, 
                                                       reps, rep_conf, stride)
            
            key = (cam_id, track_id)
            self._rep_overlay[key] = {
                "exercise": exercise_entry.exercise,
                "exercise_conf": exercise_entry.confidence,
                "reps": reps,
                "conf": rep_conf,
                "ts": time.time(),
                "voting_cycle_id": exercise_entry.voting_cycle_id,
                "entry_index": entry_index
            }
            
            updated_entry = self.exercise_tracker.get_track_history(cam_id, track_id)[entry_index]
            self.log_to_csv(updated_entry, cam_id, track_id, entry_index)
            
        except Exception as e:
            pass

    def _init_redis_connection(self):
        """Initialize Redis connection for SMPL communication."""
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

    def _start_embedding_consumer(self):
        """Start the Redis embedding consumer thread."""
        self.embedding_consumer_running = True
        self.embedding_consumer_thread = threading.Thread(
            target=self._embedding_consumer_loop,
            daemon=True
        )
        self.embedding_consumer_thread.start()

    def _embedding_consumer_loop(self):
        """
        Consumer loop that listens for new embeddings from the embedding collector.

        Runs in a background thread and processes new embeddings as they arrive
        via Redis Streams, adding them to the gallery immediately.
        """
        # Track last ID read from stream
        last_id = '0'  # Start from beginning on first run

        while self.embedding_consumer_running:
            try:
                if self.redis_client is None:
                    time.sleep(1)
                    continue

                # Block for up to 1 second waiting for new messages
                # This allows the thread to check embedding_consumer_running periodically
                messages = self.redis_client.xread(
                    {REDIS_EMBEDDING_STREAM: last_id},
                    count=10,
                    block=1000  # 1 second timeout
                )

                if not messages:
                    continue

                # Process each message
                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        try:
                            # Deserialize the embedding data
                            serialized_data = message_data.get(b'data')
                            if serialized_data is None:
                                continue

                            payload = pickle.loads(serialized_data)

                            person_name = payload.get('person_name')
                            face_id = payload.get('face_id')
                            embeddings = payload.get('embeddings')
                            visibility = payload.get('visibility')
                            pids = payload.get('pids')

                            if embeddings is None or visibility is None or pids is None:
                                continue

                            # Move tensors to device
                            embeddings = embeddings.to(self.device)
                            visibility = visibility.to(self.device)
                            pids = pids.to(self.device)

                            # Add to gallery
                            success = self.add_to_gallery(embeddings, visibility, pids)

                            if success:
                                # Store mapping from numeric PID to person display name
                                numeric_pid = int(pids[0].item())
                                self.pid_to_user_id[numeric_pid] = person_name if person_name else face_id

                            else:
                                pass

                        except Exception as e:
                            pass

                        # Update last_id to acknowledge this message
                        last_id = message_id.decode() if isinstance(message_id, bytes) else message_id

            except Exception as e:
                time.sleep(1)  # Brief pause before retrying


    def _stop_embedding_consumer(self):
        """Stop the Redis embedding consumer thread."""
        self.embedding_consumer_running = False
        if self.embedding_consumer_thread and self.embedding_consumer_thread.is_alive():
            self.embedding_consumer_thread.join(timeout=3.0)
            if self.embedding_consumer_thread.is_alive():
                pass
            else:
                pass

    def buffer_frame_for_smpl(self, frame, reid_result, camera_id, track_id):
        """
        Buffer full frame for SMPL, synchronized with MViT's 64-frame window logic.

        This mirrors MViT's global_counter system:
        - Every 64 frames creates a new window
        - Frames are stored indexed by global_counter
        - When voting result arrives, we retrieve frames by global_counters

        OPTIMIZATION: We only buffer every 5th frame (matching our sampling rate)
        This reduces memory by 5x and avoids the frame overwrite issue.

        Args:
            frame: Full video frame (not cropped)
            reid_result: ReID result for this person
            camera_id: Camera identifier
            track_id: Track identifier
        """
        key = (camera_id, int(track_id))

        # Initialize counters for this track if not exists
        if key not in self.smpl_track_frame_counter:
            self.smpl_track_frame_counter[key] = 0
            self.smpl_track_window_counter[key] = 0

        # Get current window's global_counter
        current_global_counter = self.smpl_track_window_counter[key]
        current_frame_idx = self.smpl_track_frame_counter[key]

        # Only buffer every 5th frame (0, 5, 10, 15, ..., 60)
        # This matches our sampling rate and reduces memory by 5x
        # We copy the frame here because camera buffer gets reused
        if current_frame_idx % 5 == 0:
            self.smpl_global_frame_buffer[key][current_global_counter].append({
                'frame': frame.copy(),  # Must copy - camera buffer gets reused
                'reid_result': reid_result.copy() if isinstance(reid_result, dict) else reid_result,
                'frame_idx': current_frame_idx
            })

        # Increment frame counter
        self.smpl_track_frame_counter[key] += 1

        # Check if we've completed a 64-frame window
        if self.smpl_track_frame_counter[key] >= 64:
            # Move to next window
            self.smpl_track_window_counter[key] += 1
            self.smpl_track_frame_counter[key] = 0

            # Clean up old windows (keep last 10 windows)
            # We need to keep more windows because voting happens after 3 windows complete
            old_counters = [gc for gc in self.smpl_global_frame_buffer[key].keys()
                          if gc < current_global_counter - 9]
            for old_gc in old_counters:
                del self.smpl_global_frame_buffer[key][old_gc]

    def send_sampled_frames_to_smpl(self, cam_id, track_id, global_counters, exercise_name, voting_cycle_id):
        """
        Send sampled frames to SMPL when voting result arrives.

        Retrieves frames from buffer using global_counters from MViT voting result,
        samples every 5th frame (0, 5, 10, 15, ...), and sends to Redis.

        Args:
            cam_id: Camera identifier
            track_id: Track identifier
            global_counters: List of 3 global counters from voting result (e.g., [5, 6, 7])
            exercise_name: Detected exercise name
            voting_cycle_id: Unique identifier for this voting cycle

        Returns:
            int: Number of frames sent
        """
        if self.redis_client is None:
            return 0

        key = (cam_id, int(track_id))

        # Check if we already sent frames for this voting cycle
        if voting_cycle_id in self.smpl_sent_for_voting_cycle:
            return 0

        # Collect all frames from the 3 windows (192 frames total)
        all_frames = []
        for gc in global_counters:
            window_frames = self.smpl_global_frame_buffer[key].get(gc, [])
            all_frames.extend(window_frames)

        if not all_frames:
            # Debug: show what we have in the buffer
            available_counters = list(self.smpl_global_frame_buffer[key].keys())
            return 0

        # Frames are already sampled (every 5th) during buffering
        # 3 windows x 64 frames / 5 = ~39 frames total
        sampled_frames = all_frames


        # Use Redis pipeline for efficient batch sending (much faster than individual xadd calls)
        try:
            pipe = self.redis_client.pipeline(transaction=False)
            frames_sent = 0

            for i, frame_data in enumerate(sampled_frames):
                frame = frame_data['frame']
                reid_result = frame_data['reid_result']

                # Convert bbox to list if needed
                bbox = reid_result.get("bbox")
                if bbox is not None and hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                elif bbox is not None:
                    bbox = list(bbox)

                similarity = reid_result.get("similarity")
                person_name = reid_result.get("person_name", "Unknown")

                payload = {
                    "frame": frame,  # Already copied during buffering
                    "camera_id": cam_id,
                    "exercise_name": exercise_name,
                    "reid_result": {
                        "track_id": int(track_id),
                        "person_name": person_name,
                        "bbox": bbox,
                        "exercise": exercise_name,
                        "reps": reid_result.get("reps"),
                        "similarity": float(similarity) if similarity is not None else None
                    },
                    "voting_cycle_id": voting_cycle_id,
                    "frame_index": i,
                    "total_frames": len(sampled_frames),
                    "global_counters": global_counters,
                    "timestamp": time.time()
                }

                # Serialize and add to pipeline
                serialized_data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                pipe.xadd(
                    REDIS_STREAM,
                    {'data': serialized_data},
                    maxlen=REDIS_MAX_QUEUE_LEN,
                    approximate=True
                )
                frames_sent += 1

            # Execute all commands in one batch
            pipe.execute()

        except Exception as e:
            frames_sent = 0

        # Mark this voting cycle as sent
        self.smpl_sent_for_voting_cycle.add(voting_cycle_id)

        # Clean up old voting cycle IDs (keep last 100)
        if len(self.smpl_sent_for_voting_cycle) > 100:
            # Convert to list, sort, remove oldest
            old_cycles = sorted(self.smpl_sent_for_voting_cycle)[:50]
            for vc in old_cycles:
                self.smpl_sent_for_voting_cycle.discard(vc)

        return frames_sent

    # ==========================================
    # WEBSOCKET METHODS (FALLBACK - COMMENTED)
    # ==========================================
    # Uncomment these methods if you want to use WebSocket instead of Redis

    # def _start_websocket_loop(self):
    #     """Start a background thread with a persistent event loop for websocket operations."""
    #     import time
    #
    #     def run_loop():
    #         self.ws_loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(self.ws_loop)
    #         self.ws_loop.run_forever()
    #
    #     self.ws_thread = threading.Thread(target=run_loop, daemon=True)
    #     self.ws_thread.start()
    #
    #     # Wait for loop to be initialized
    #     max_wait = 10  # Maximum 1 second wait
    #     for _ in range(max_wait):
    #         if self.ws_loop is not None:
    #             break
    #         time.sleep(0.1)
    #
    #     if self.ws_loop is None:
    #         print("[WS] ERROR: Failed to initialize websocket event loop")
    #         return
    #
    #     print("[WS] Websocket event loop started in background thread")
    #
    #     # Connect to websocket immediately on startup
    #     future = asyncio.run_coroutine_threadsafe(self.connect_websocket(), self.ws_loop)
    #     try:
    #         future.result(timeout=5.0)
    #     except Exception as e:
    #         print(f"[WS] Initial connection failed: {e}")
    #
    # async def connect_websocket(self):
    #     """Connect to the websocket server."""
    #     try:
    #         print(f"[WS] Connecting to websocket server at {WS_URL}...")
    #         self.ws_connection = await websockets.connect(WS_URL, max_size=None)
    #         print("[WS] ✓ Websocket connection established!")
    #         return True
    #     except Exception as e:
    #         print(f"[WS] Failed to connect to websocket: {e}")
    #         self.ws_connection = None
    #         return False
    #
    # async def ensure_websocket_connected(self):
    #     """Ensure websocket is connected, attempt reconnection if needed."""
    #     # Only reconnect if connection was lost
    #     if self.ws_connection is None:
    #         with self.ws_lock:
    #             if self.ws_connection is None:
    #                 print("[WS] Connection lost, attempting to reconnect...")
    #                 return await self.connect_websocket()
    #     # Connection exists, assume it's good
    #     return True
    #
    # async def send_exercise_to_smpl(self, frame, exercise_name, reid_result, camera_id, entry_index=None):
    #     """
    #     Send frame and exercise data to SMPL server when exercise is detected.
    #     Only sends every 5th frame to reduce bandwidth.
    #
    #     Args:
    #         frame: Video frame to send
    #         exercise_name: Name of the detected exercise
    #         reid_result: The reid result containing the person with exercise detection
    #         camera_id: Camera identifier
    #         entry_index: Index of the exercise entry in the track history (optional)
    #     """
    #     try:
    #         # Increment frame counter
    #         self.ws_frame_counter[camera_id] += 1
    #
    #         # Only send every 5th frame
    #         if self.ws_frame_counter[camera_id] % 5 != 0:
    #             return False
    #
    #         # Ensure websocket is connected
    #         if not await self.ensure_websocket_connected():
    #             print("[WS] Failed to establish websocket connection")
    #             return False
    #
    #         # Resize frame for efficient transmission
    #         # resized_frame = cv2.resize(frame, (640, 480))
    #
    #         # Encode frame as JPEG
    #         success, buffer = cv2.imencode(
    #             ".jpg",
    #             frame,
    #             [cv2.IMWRITE_JPEG_QUALITY, 75]
    #         )
    #
    #         # if not success:
    #         #     print("[WS] Failed to encode frame as JPEG")
    #         #     return False
    #
    #         # Base64 encode the image
    #         jpg_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    #
    #         # Prepare payload with frame, camera_id, exercise, and reid_result
    #         # Convert bbox to list if it's a numpy array
    #         bbox = reid_result.get("bbox")
    #         if bbox is not None and hasattr(bbox, 'tolist'):
    #             bbox = bbox.tolist()
    #         elif bbox is not None:
    #             bbox = list(bbox)
    #
    #         # Get values from reid_result (already processed correctly)
    #         similarity = reid_result.get("similarity")
    #         person_name = reid_result.get("person_name", "Unknown")
    #         track_id = reid_result.get("track_id")
    #
    #         payload = {
    #             "frame": jpg_base64,
    #             "camera_id": camera_id,
    #             "exercise_name": exercise_name,
    #             "reid_result": {
    #                 "track_id": int(track_id) if track_id is not None else None,
    #                 "person_name": person_name,
    #                 "bbox": bbox,
    #                 "exercise": reid_result.get("exercise"),
    #                 "reps": reid_result.get("reps"),
    #                 "similarity": float(similarity) if similarity is not None else None
    #             },
    #             "entry_index": entry_index,
    #             "timestamp": time.time()
    #         }
    #
    #         # Convert to JSON
    #         json_payload = json.dumps(payload)
    #
    #         # Debug: Print payload info
    #         print(f"person_name {person_name} (similarity: {similarity})")
    #         # Send payload through persistent websocket connection
    #         if self.ws_connection is not None:
    #             await self.ws_connection.send(json_payload)
    #             print(f"[WS] Sent frame {self.ws_frame_counter[camera_id]} from {camera_id} - Exercise: {exercise_name} - Track: {track_id} - Person: {person_name}")
    #             return True
    #         else:
    #             print("[WS] Websocket connection is None")
    #             return False
    #
    #     except Exception as e:
    #         print(f"[WS] Error in send_exercise_to_smpl: {e}")
    #         # Mark connection as broken so it will reconnect on next attempt
    #         self.ws_connection = None
    #         return False
    #
    # async def close_websocket(self):
    #     """Close the websocket connection gracefully with timeout."""
    #     if self.ws_connection is not None:
    #         try:
    #             await asyncio.wait_for(self.ws_connection.close(), timeout=2.0)
    #             print("[WS] Websocket connection closed")
    #         except asyncio.TimeoutError:
    #             print("[WS] Websocket close timed out, forcing close")
    #         except Exception as e:
    #             print(f"[WS] Error closing websocket: {e}")
    #         finally:
    #             self.ws_connection = None

    def draw_reid_results(self, frame, reid_results, camera_id):
        """
        Draw person re-identification results with exercise and weight information on frame.

        Args:
            frame (numpy.ndarray): Video frame to draw on
            reid_results (list): List of person detection results
            camera_id (str): Camera identifier

        Returns:
            numpy.ndarray: Annotated frame with drawn results (or original frame if visual_mode=False)
        """
        # If visual mode is disabled, skip all drawing operations
        if not self.visual_mode:
            return frame

        now = time.time()

        # Augment reid_results with exercise/reps from overlay
        # Note: SMPL frames are now sent via process_voting_results() when voting completes,
        # not here in draw_reid_results. This removes the bottleneck.
        for r in reid_results or []:
            tid = r.get("track_id")
            if tid is None:
                continue
            key = (camera_id, int(tid))

            overlay = self._rep_overlay.get(key)
            if overlay and (now - overlay["ts"] < 15.0):
                exercise = overlay.get("exercise")
                exercise_conf = overlay.get("exercise_conf", 0.0)
                reps = overlay.get("reps")
                rep_conf = overlay.get("conf", 0.0)

                if exercise_conf >= self.ex_conf_threshold:
                    r["exercise"] = f"{exercise} ({exercise_conf:.2f})"
                    if reps is not None:
                        r["reps"] = f"{int(reps)} reps ({rep_conf:.2f})"
                        summary = self.exercise_tracker.get_track_summary(camera_id, tid)
                        r["master_info"] = f"Total: {summary['total_exercises']}"

        # Draw weights on frame
        w_dets = self.last_weight_detections.get(camera_id, [])
        if w_dets:
            for d in w_dets:
                x1, y1, x2, y2 = d['bbox']
                label = d['label']
                conf_emb = d['conf_emb']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                txt = f"{label} ({conf_emb:.2f})"
                tsize = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                tx, ty = x1, max(0, y1 - 10)
                cv2.rectangle(frame, (tx, ty - tsize[1] - 5), (tx + tsize[0] + 10, ty + 5), (255, 255, 0), -1)
                cv2.putText(frame, txt, (tx + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        annotated_frame = super().draw_reid_results(frame, reid_results, camera_id)
        return annotated_frame

    def stop_system(self):
        """
        Shutdown system with proper cleanup.

        Clears weight detection cache and calls parent class cleanup methods.
        """
        # Stop embedding consumer thread first
        self._stop_embedding_consumer()

        # Close database connection pool
        from utils.database_manager import DatabaseManager
        DatabaseManager.close_all_connections()

        # Close Redis connection
        if USE_REDIS and self.redis_client is not None:
            try:
                self.redis_client.close()
            except Exception as e:
                pass

        # Clear SMPL buffers
        self.smpl_global_frame_buffer.clear()
        self.smpl_sent_for_voting_cycle.clear()

        # Close websocket connection (COMMENTED - uncomment if using WebSocket)
        # if self.ws_connection is not None:
        #     try:
        #         if self.ws_loop is not None and self.ws_loop.is_running():
        #             future = asyncio.run_coroutine_threadsafe(self.close_websocket(), self.ws_loop)
        #             future.result(timeout=5.0)
        #     except Exception as e:
        #         print(f"[WS] Error during websocket cleanup: {e}")
        #
        # # Stop the websocket event loop
        # if self.ws_loop is not None:
        #     self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
        #     if self.ws_thread is not None:
        #         self.ws_thread.join(timeout=2.0)

        self.last_weight_detections.clear()
        if hasattr(super(), 'stop_system'):
            super().stop_system()

def check_database_connection():
    """
    Check database connection and setup tables.

    Performs database connectivity test, creates necessary tables,
    and validates that required data is present.

    Returns:
        bool: True if database is properly connected and configured, False otherwise
    """
    global database_checked
    if database_checked:
        return True
    try:
        db_manager = DatabaseManager()
        
        connected, message = db_manager.test_connection()
        if not connected:
            return False
        
        
        # Create exercise_logs table (legacy support)
        db_manager.create_exercise_logs_table()
        
        stats = db_manager.get_database_stats()
        
        if stats['person_embeddings'] == 0:
            if os.getenv('HEADLESS'):
                return True
            else:
                response = input("Continue anyway? (y/n): ").lower()
                if response != 'y':
                    return False
        else:
            pass
        
        database_checked = True
        return True
        
    except Exception as e:
        return False
    

def main():
    """
    Main function to initialize and run the face/person re-identification system.

    Sets up SlowFast exercise recognition engine, RepNet repetition counting engine,
    camera configurations, database connections, and starts the complete system.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # MViT config
    weights_path = "models/52_class_model_acc84_29_12_25.pt"

    # class_names = [
    #     "Bicep Curl", "Front Raise", "Hammer Curl", "Lateral Raise", "Upright Rows",
    #     "DB Chest Press", "DB Incline Chest Press", "DB Lunges",
    #     "DB Reverse Flys", "KB Goblet Squat", "KB Overhead Press", "KB Swings",
    #     "KB Goodmorning", "Seated DB Shoulder Pess", "Single Arm DB Row"
    # ]
    class_names = [
    "Banded_curls", "arm_circles", "bb_military_press",
    "bb_upright", "bench_ab_crunch", "bicep_curls",
    "bus_driver", "chest_rows", "concentration_curls",
    "db_chest_fly", "db_chest_press", "db_hip_thrust",
    "db_incline_chest_press", "db_lunges", "db_reverse_flys",
    "db_rows", "db_seated_arnoldpress", "db_seated_hammercurl",
    "db_seated_shoulder_press", "db_tricep_kickback", "db_upright_rows",
    "double_kb_squat", "ez_bb_curls", "front_raise",
    "glute_kick_back", "hammer_curls", "kb_clean and press",
    "kb_floor_press", "kb_goblet_squats", "kb_goodmorning",
    "kb_halo", "kb_overhead_press", "kb_shrugs",
    "kb_snatch", "kb_swings", "kb_windmill",
    "lateral_raise", "plank_lying", "renegade_rows_db",
    "romanian_deadlift_db", "russian_twist", "seated_db_low to high_fly",
    "side_plank", "singlearm_dumbbell_rows", "skull_crushers",
    "spider_curls", "standing_arnold_press", "standing_banded_curls",
    "sumo_squat_db", "superman_lying", "tricep_dips","zottman_curls"
]
    
    if not os.path.exists(weights_path):
        return 1

    sf = MViTEngine(
        class_names=class_names,
        num_frames=16,
        max_microbatch=24,  # Increased from 12 to 24 for better GPU utilization with 15 cameras
        tick_ms=120,
        cooldown_s=0.4,
        model_path=weights_path,
        num_classes=len(class_names),
        K=5,
        use_fp16=False,  # Disabled FP16 to match working MVit_inference_rtsp.py
        confidence_threshold=0.3,  
    )

    sf.start()
    time.sleep(5)
    init_stats = sf.get_stats()
    if not init_stats.get('model_loaded', False):
        return 1

    # RepNet config
    repnet_weights_path = "models/repnet.pth"
    repnet = None
    if os.path.isfile(repnet_weights_path):
        repnet = RepNetEngine(
            weights_path=repnet_weights_path,
            device="cuda",
            default_stride=3,
            input_size=112,  
        )
    
    DEFAULT_CAMERA_IPS = ["192.168.0.130"]

    if not check_database_connection():
        return 1

    camera_ips = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CAMERA_IPS
    camera_configs = create_camera_configs_from_ips(camera_ips)

    for i, cfg in enumerate(camera_configs, 1):
        pass

    # BPBReID configuration
    BPBREID_CONFIG_PATH = "configs/test_reid.yaml"
    BPBREID_WEIGHTS_PATH = None  # Will use weights from config file
    BPBREID_GALLERY_DIR = None  # Will load from database dynamically
    VOTING_THRESHOLD = 6.5  # Distance < threshold counts as a vote
    VOTING_WINDOW = 40
    MIN_VOTES = 30
    MATCHING_THRESHOLD = 6.7  # Distance < threshold shows as recognized
    DEVICE = 'cuda'

    # Initialize BridgeReIDService with BPBReID
    reid = BridgeReIDService(
        stream_configs=camera_configs,
        config_path=BPBREID_CONFIG_PATH,
        weights_path=BPBREID_WEIGHTS_PATH,
        gallery_dir=BPBREID_GALLERY_DIR,
        voting_threshold=VOTING_THRESHOLD,
        voting_window=VOTING_WINDOW,
        min_votes=MIN_VOTES,
        matching_threshold=MATCHING_THRESHOLD,
        device=DEVICE,
        mvit_engine=sf,
        repnet_engine=repnet,
        visual_mode=VISUAL_MODE
    )

    def _shutdown(*_):
        try:
            reid.stop_system()
        except Exception as e:
            pass
        sf.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        reid.start_person_reid_system()
        return 0
    except Exception as e:
        return 1
    finally:
        sf.stop()

if __name__ == "__main__":
    sys.exit(main())
    
