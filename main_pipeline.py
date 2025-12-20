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
from services.enhanced_slowfast_engine import SlowFastEngine
from services.person_reid_service import PersonReIDService
from services.RepNet_engine import RepNetEngine
from services.pose_yolo import PoseYolo
from camera import create_camera_configs_from_ips
from utils.database_manager import DatabaseManager
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from keypoint_processing import integrate_keypoint_processor
# Add to imports
from pose_analysis import init_pose_analyzer, analyse_pose, get_pose_analysis_stats


database_checked = False

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
            print(f"[MASTER_LIST] Added {cam_id}:{track_id} #{entry_index}: {exercise_entry.exercise} "
                  f"({exercise_entry.confidence:.3f}) - {len(exercise_entry.frames)} frames")
            
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
                    print(f"[REPNET_SEND] {cam_id}:{track_id} - Sending exercise #{i}: {entry.exercise} to RepNet")
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
                print(f"[REPNET_DONE] {cam_id}:{track_id} #{entry_index}: "
                    f"{entry.exercise} -> {reps} reps (conf: {rep_conf:.3f})")

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
    
    def __init__(self, *args, slowfast_engine, repnet_engine=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_buf = defaultdict(list)        # (cam,tid) -> list of 224x224 RGB frames
        self.pose_emit_idx = defaultdict(int)    # (cam,tid) -> cursor

        self.sf = slowfast_engine
        self.repnet = repnet_engine
        self.enable_repnet = self.repnet is not None
        self.enable_slowfast = self.sf is not None
        
        self.exercise_tracker = global_exercise_tracker
        self._rep_overlay = {}  # (cam,tid) -> latest RepNet result for display
        self.ex_conf_threshold = 0.7

        # Initialize database manager
        self.db_manager = DatabaseManager()

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

        self.pose_batch_csvpath = "pose_batches.csv"
        with open(self.pose_batch_csvpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "camera_id", "track_id", "win_idx", 
                            "seq_start", "seq_end", "avg_pose_score", "total_frames"])

        self.pose_keypoints_csvpath = "pose_keypoints.csv" 
        with open(self.pose_keypoints_csvpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "camera_id", "track_id", "win_idx", 
                            "frame_rel_idx", "keypoints_json", "pose_score"])

        # Weight detection
        self.weight_engine = None
        self.weight_detection_stride = 1
        self.frame_counters = {}
        self.last_weight_detections = {}
        self.init_weight_engine()
        
        # Initialize pose model
        self.pose_model = PoseYolo(weights="models/yolo11n-pose.pt", conf=0.25, iou=0.45)

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
                print("[WEIGHT] Weight recognition engine initialized")
            else:
                print("[WEIGHT] Weight model files not found, skipping weight detection")
                self.weight_engine = None
        except Exception as e:
            print(f"[WEIGHT] Failed to initialize weight engine: {e}")
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
                print(f"[DB] Inserted exercise log: {cam_id}:{track_id} - {entry.exercise} (reps: {entry.reps})")
                
        except Exception as db_error:
            print(f"[DB_ERROR] Exercise database logging failed: {db_error}")

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
                key = (cam, tid)
                self.pose_buf[key].append(rgb224)
                
                # Pose 64-frame windowing with database integration
                p_start = self.pose_emit_idx.setdefault(key, 0)
                while len(self.pose_buf[key]) - p_start >= 64:
                    window = self.pose_buf[key][p_start:p_start+64]
                    win_idx = p_start // 64

                    # Run YOLO-Pose on the 64 frames
                    try:
                        pose_out = self.pose_model.infer_frames(window)
                    except Exception as e:
                        pose_out = [{"keypoints": [], "score": 0.0} for _ in range(64)]

                    # Process pose data
                    try:
                        current_time = datetime.now().isoformat()
                        
                        # Calculate batch statistics
                        scores = [po.get("score", 0.0) for po in pose_out]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        
                        # Database insertion - Pose batch
                        pose_batch_data = {
                            'timestamp': current_time,
                            'camera_id': cam,
                            'track_id': tid,
                            'win_idx': win_idx,
                            'seq_start': p_start + 1,
                            'seq_end': p_start + 64,
                            'avg_pose_score': avg_score,
                            'total_frames': len(pose_out)
                        }
                        
                        batch_success = self.db_manager.insert_pose_batch(pose_batch_data)
                        if batch_success:
                            print(f"[DB_POSE] Inserted pose batch: {cam}:{tid} win_idx={win_idx}")
                        
                        # Database insertion - Pose keypoints (batch)
                        keypoints_batch = []
                        for rel_idx, po in enumerate(pose_out):
                            keypoints_batch.append({
                                'timestamp': current_time,
                                'camera_id': cam,
                                'track_id': tid,
                                'win_idx': win_idx,
                                'frame_rel_idx': rel_idx,
                                'keypoints_json': {"kps": po.get("keypoints", [])},
                                'pose_score': float(po.get("score", 0.0))
                            })
                        
                        keypoints_success = self.db_manager.batch_insert_pose_keypoints(keypoints_batch)
                        if keypoints_success:
                            print(f"[DB_POSE] Inserted {len(keypoints_batch)} keypoints for {cam}:{tid} win_idx={win_idx}")
                        
                        # CSV backup logging
                        with open(self.pose_batch_csvpath, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                current_time, cam, tid, win_idx,
                                p_start + 1, p_start + 64, f"{avg_score:.3f}", len(pose_out)
                            ])
                        
                        with open(self.pose_keypoints_csvpath, "a", newline="") as f:
                            writer = csv.writer(f)
                            for rel_idx, po in enumerate(pose_out):
                                keypoints_json = json.dumps({"kps": po.get("keypoints", [])})
                                score = float(po.get("score", 0.0))
                                writer.writerow([
                                    current_time, cam, tid, win_idx, rel_idx,
                                    keypoints_json, f"{score:.3f}"
                                ])
                        
                    except Exception as e:
                        print(f"[POSE_ERROR] {cam}:{tid} win_idx={win_idx} processing failed: {e}")

                    p_start += 64

                self.pose_emit_idx[key] = p_start

                # CLEANED: Only submit 224x224 RGB crop to SlowFast
                if tid >= 0 and self.enable_slowfast:
                    self.sf.submit_crop(cam, tid, rgb224)
                    
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
                                    print(f"[DB_WEIGHT] {camera_id}:{track_id} -> {label} (conf: {conf:.3f})")
                                
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
                print(f"[WEIGHT] Detection failed: {e}")
                self.last_weight_detections[camera_id] = []
        else:
            weight_detections = self.last_weight_detections.get(camera_id, [])

        # Process SlowFast voting results
        if self.enable_slowfast and self.enable_repnet:
            self.process_voting_results()

        return results

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
                    
                    self.exercise_tracker.add_exercise_entry(cam, tid, exercise_entry)
            
            self.process_pending_repnet_entries()
                        
        except Exception as e:
            print(f"[VOTING_ERROR] Error processing voting results: {e}")

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
                "voting_cycle_id": exercise_entry.voting_cycle_id
            }
            
            updated_entry = self.exercise_tracker.get_track_history(cam_id, track_id)[entry_index]
            self.log_to_csv(updated_entry, cam_id, track_id, entry_index)
            
        except Exception as e:
            print(f"[REPNET_ERROR] {cam_id}:{track_id} - {e}")

    def draw_reid_results(self, frame, reid_results, camera_id):
        """
        Draw person re-identification results with exercise and weight information on frame.

        Args:
            frame (numpy.ndarray): Video frame to draw on
            reid_results (list): List of person detection results
            camera_id (str): Camera identifier

        Returns:
            numpy.ndarray: Annotated frame with drawn results
        """
        now = time.time()

        # Augment reid_results with exercise/reps
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
            print(f"[ERROR] Database connection failed: {message}")
            return False
        
        print(f"[SUCCESS] {message}")
        
        # Create exercise_logs table (legacy support)
        db_manager.create_exercise_logs_table()
        
        stats = db_manager.get_database_stats()
        
        if stats['person_embeddings'] == 0:
            print("WARNING: No person embeddings found in database!")
            if os.getenv('HEADLESS'):
                print("Running in headless mode, continuing without embeddings...")
                return True
            else:
                response = input("Continue anyway? (y/n): ").lower()
                if response != 'y':
                    return False
        else:
            print(f"Found {stats['person_people']} people with {stats['person_embeddings']} body embeddings")
        
        database_checked = True
        return True
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False
    

def main():
    """
    Main function to initialize and run the face/person re-identification system.

    Sets up SlowFast exercise recognition engine, RepNet repetition counting engine,
    camera configurations, database connections, and starts the complete system.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # SlowFast config
    #  weights_path = "models/E60_C20_Cam10718.pt"
    weights_path = "models/slowfast_15class.pt"

    class_names = [
        "Bicep Curl", "Front Raise", "Hammer Curl", "Lateral Raise", "Upright Rows",
        "DB Chest Press", "DB Incline Chest Press", "DB Lunges",
        "DB Reverse Flys", "KB Goblet Squat", "KB Overhead Press", "KB Swings",
        "KB Goodmorning", "Seated DB Shoulder Pess", "Single Arm DB Row"
    ]

   
    if not os.path.exists(weights_path):
        print(f"ERROR: Custom weights file not found: {weights_path}")
        return 1

    sf = SlowFastEngine(
        class_names=class_names,
        t_fast=32,
        alpha=4,
        max_microbatch=12,
        tick_ms=120,
        cooldown_s=0.4,
        model_name="slowfast_r50",
        weights_path=weights_path,
        use_fp16=True,
    )

    print("Starting SlowFast engine...")
    sf.start()
    time.sleep(5)
    init_stats = sf.get_stats()
    if not init_stats.get('model_loaded', False):
        print("ERROR: SlowFast model failed to load!")
        return 1
    print("SUCCESS: SlowFast model loaded!")

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
        print("SUCCESS: RepNet model loaded!")
    
    # DEFAULT_CAMERA_IPS = ['192.168.0.120',"192.168.0.196","192.168.0.114",
    #                       "192.168.0.18","192.168.0.110","192.168.0.102",
    #                       "192.168.0.124","192.168.0.123","192.168.0.106",
    #                       "192.168.0.113","192.168.0.111","192.168.0.117","192.168.0.112"]
    DEFAULT_CAMERA_IPS = ["192.168.0.124","192.168.0.216","192.168.0.110"]

    if not check_database_connection():
        return 1

    camera_ips = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CAMERA_IPS
    camera_configs = create_camera_configs_from_ips(camera_ips)

    print(f"Using {len(camera_configs)} cameras:")
    for i, cfg in enumerate(camera_configs, 1):
        print(f"  {i}: {cfg['name']} - {cfg['url']}")

    # BPBreID configuration (same as rtsp_reid_inference.py)
    BPBREID_CONFIG_PATH = "configs/test_reid.yaml"
    BPBREID_WEIGHTS_PATH = None  # Will use weights from config file
    BPBREID_GALLERY_DIR = None  # Will load from database dynamically
    VOTING_THRESHOLD = 4.0
    VOTING_WINDOW = 50
    MIN_VOTES = 30
    MATCHING_THRESHOLD = 5.0
    DEVICE = 'cuda'

    # Initialize BridgeReIDService with BPBreID
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
        slowfast_engine=sf,
        repnet_engine=repnet
    )
    print("[POSE_ANALYSIS] Initializing pose analyzer...")
    pose_analyzer = init_pose_analyzer(reid.db_manager)
    print("[POSE_ANALYSIS] Pose analyzer initialized")
    
    # Start keypoint processor with real analysis function
    print("[KEYPOINT] Starting continuous keypoint processor...")
    keypoint_processor = integrate_keypoint_processor(
        db_manager=reid.db_manager,
        analyse_pose_func=analyse_pose,  # Now uses the real analysis function
        processing_interval=9
    )
    print("[KEYPOINT] Keypoint processor started")
    def _shutdown(*_):
        print("Shutting down...")
        try:
            reid.stop_system()
        except Exception as e:
            print(f"Error stopping ReID system: {e}")
        sf.stop()
        print("System stopped")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        print("Starting person ReID system")
        reid.start_person_reid_system()
        return 0
    except Exception as e:
        print(f"Error starting ReID system: {e}")
        return 1
    finally:
        sf.stop()

if __name__ == "__main__":
    sys.exit(main())
    

