
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
import logging
import psycopg2.extras as extras
from collections import defaultdict

logger = logging.getLogger(__name__)

class ContinuousKeypointProcessor:
    """
    Continuous processor that monitors exercise logs and matches them with keypoint data
    Sends matched data to analyse_pose() function instead of saving to CSV
    """
    
    def __init__(self, db_manager, analyse_pose_func: Callable, processing_interval: int = 30):
        """
        Initialize the continuous processor
        
        Args:
            db_manager: DatabaseManager instance
            analyse_pose_func: Function to call with matched exercise-keypoint data
            processing_interval: How often to check for new exercise logs (seconds)
        """
        self.db_manager = db_manager
        self.analyse_pose = analyse_pose_func  # Function to call with results
        self.processing_interval = processing_interval
        
        # Threading control
        self._running = False
        self._processor_thread = None
        self._lock = threading.Lock()
        
        # Track last processed exercise log ID to avoid reprocessing
        self._last_processed_id = 0
        
        # Statistics
        self.stats = {
            'total_exercise_logs_processed': 0,
            'total_matched_records': 0,
            'total_analyse_calls': 0,
            'processing_cycles': 0,
            'last_processing_time': None,
            'errors': 0
        }
        
    def start_continuous_processing(self):
        """Start the continuous processing thread"""
        if self._running:
            logger.warning("Continuous processing already running")
            return
            
        logger.info("Starting continuous keypoint processing...")
        self._running = True
        self._processor_thread = threading.Thread(
            target=self.processing_loop,
            name="KeypointProcessor",
            daemon=True
        )
        self._processor_thread.start()
        logger.info("Continuous keypoint processing started")
        
    def stop_continuous_processing(self):
        """Stop the continuous processing"""
        if not self._running:
            return
            
        logger.info("Stopping continuous keypoint processing...")
        self._running = False
        
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=10)
            
        logger.info("Continuous keypoint processing stopped")
    
    def processing_loop(self):
        """Main processing loop that runs continuously"""
        logger.info(f"Processing loop started - checking every {self.processing_interval} seconds")
        
        while self._running:
            try:
                cycle_start = time.time()
                
                # Process new exercise logs
                processed_count = self.process_new_exercise_logs()
                
                with self._lock:
                    self.stats['processing_cycles'] += 1
                    self.stats['last_processing_time'] = datetime.now().isoformat()
                
                cycle_duration = time.time() - cycle_start
                
                if processed_count > 0:
                    logger.info(f"Processing cycle completed: {processed_count} logs processed in {cycle_duration:.2f}s")
                
                # Sleep until next cycle
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                with self._lock:
                    self.stats['errors'] += 1
                time.sleep(self.processing_interval)  # Continue despite errors
    
    def process_new_exercise_logs(self) -> int:
        """Process new exercise logs since last check"""
        try:
            # Get new exercise logs since last processed ID
            new_logs = self.fetch_new_exercise_logs()
            
            if not new_logs:
                return 0
                
            logger.info(f"Found {len(new_logs)} new exercise logs to process")
            
            # Group by camera/track for processing
            grouped_logs = self.group_logs_by_track(new_logs)
            
            total_processed = 0
            
            for (camera_id, track_id), logs in grouped_logs.items():
                try:
                    matched_count = self.process_track_logs(camera_id, track_id, logs)
                    total_processed += matched_count
                    
                except Exception as e:
                    logger.error(f"Error processing logs for {camera_id}:{track_id}: {e}")
                    continue
            
            # Update last processed ID
            if new_logs:
                max_id = max(log['id'] for log in new_logs)
                self._last_processed_id = max_id
                
            with self._lock:
                self.stats['total_exercise_logs_processed'] += len(new_logs)
            
            return total_processed
            
        except Exception as e:
            logger.error(f"Error processing new exercise logs: {e}")
            return 0
    
    def fetch_new_exercise_logs(self) -> List[Dict]:
        """Fetch new exercise logs from database"""
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Could not get database connection")
                return []
            
            # Register JSONB handler
            extras.register_default_jsonb(conn, loads=json.loads)
            
            query = """
                SELECT id, timestamp, camera_id, track_id, exercise, exercise_conf,
                       reps, rep_conf, frame_count, voting_cycle_id, vote_counts,
                       batches_used, batch_ids, global_counters, entry_index, weight, weight_conf
                FROM exercise_logs
                WHERE id > %s
                ORDER BY timestamp, id
            """
            
            df = pd.read_sql(query, conn, params=[self._last_processed_id])
            conn.close()
            
            # Convert to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error fetching new exercise logs: {e}")
            return []
    
    def group_logs_by_track(self, logs: List[Dict]) -> Dict[tuple, List[Dict]]:
        """Group logs by camera_id and track_id"""
        grouped = defaultdict(list)
        
        for log in logs:
            key = (log['camera_id'], log['track_id'])
            grouped[key].append(log)
        
        return dict(grouped)
    
    def process_track_logs(self, camera_id: str, track_id: int, logs: List[Dict]) -> int:
        """Process logs for a specific track and match with keypoints"""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(logs)
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp_dt').reset_index(drop=True)
            
            if df.empty:
                return 0
            
            # Match with keypoints directly without set identification
            matched_data = self.match_with_keypoints(df, camera_id, track_id)
            
            if matched_data.empty:
                return 0
            
            # Process each matched exercise individually
            self.process_matched_data_individually(matched_data)
            
            with self._lock:
                self.stats['total_matched_records'] += len(matched_data)
            
            return len(matched_data)
            
        except Exception as e:
            logger.error(f"Error processing track logs for {camera_id}:{track_id}: {e}")
            return 0
    
    def parse_global_counters(self, global_counters_data) -> List[int]:
        """Parse global_counters JSONB data to list of integers"""
        try:
            if global_counters_data is None:
                return []
            
            # If it's already a list (expected with JSONB handler registered)
            if isinstance(global_counters_data, list):
                return [int(x) for x in global_counters_data]
            
            # If it's a string representation (fallback case)
            if isinstance(global_counters_data, str):
                cleaned_str = global_counters_data.strip()
                if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
                    parsed = json.loads(cleaned_str)
                    return [int(x) for x in parsed]
            
            return []
            
        except Exception as e:
            logger.warning(f"Error parsing global_counters {repr(global_counters_data)}: {e}")
            return []
    
    def match_with_keypoints(self, df: pd.DataFrame, camera_id: str, track_id: int) -> pd.DataFrame:
        """Match exercise data with keypoint data using global_counters from exercise logs"""
        try:
            all_matched_data = []
            
            for idx, exercise_row in df.iterrows():
                # Parse global_counters from the exercise log
                global_counters = self.parse_global_counters(exercise_row['global_counters'])
                
                if not global_counters:
                    logger.warning(f"No global counters for exercise row {idx} ({camera_id}/{track_id})")
                    continue
                
                # Fetch ONLY keypoints matching the global_counters (win_idx values) for this specific exercise
                keypoint_data = self.fetch_keypoint_data_for_counters(global_counters, camera_id, track_id)
                
                if keypoint_data.empty:
                    logger.warning(f"No keypoint data found for {camera_id}/{track_id} with global_counters: {global_counters}")
                    continue
                
                logger.info(f"[KEYPOINT_MATCH] Exercise log {exercise_row['id']}: Found {len(keypoint_data)} keypoint frames for {len(global_counters)} windows (expected ~{len(global_counters)*64} frames)")
                
                # Create matched records for this specific exercise log
                for _, kp_row in keypoint_data.iterrows():
                    matched_record = {
                        'camera_id': camera_id,
                        'track_id': track_id,
                        'exercise_log_id': exercise_row['id'],
                        'exercise': exercise_row['exercise'],
                        'win_idx': kp_row['win_idx'],
                        'frame_rel_idx': kp_row['frame_rel_idx'],
                        'timestamp_exercise': exercise_row['timestamp'],
                        'keypoints_json': kp_row['keypoints_json'],
                        'pose_score': kp_row['pose_score'],
                        'exercise_conf': exercise_row.get('exercise_conf', None),
                        'reps': exercise_row.get('reps', None),
                        'rep_conf': exercise_row.get('rep_conf', None),
                        'global_counters_used': global_counters  # Store which global_counters were used
                    }
                    all_matched_data.append(matched_record)
            
            if not all_matched_data:
                return pd.DataFrame()
            
            result_df = pd.DataFrame(all_matched_data)
            
            # Sort by exercise log id and frame order
            result_df = result_df.sort_values([
                'camera_id', 'track_id', 'exercise_log_id', 
                'win_idx', 'frame_rel_idx'
            ]).reset_index(drop=True)
            
            logger.debug(f"Matched {len(result_df)} keypoint records for {camera_id}:{track_id}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error matching with keypoints: {e}")
            return pd.DataFrame()
    
    def fetch_keypoint_data_for_counters(self, global_counters: List[int], camera_id: str, track_id: int) -> pd.DataFrame:
        """Fetch keypoint data for specific global counters (win_idx values)"""
        if not global_counters:
            return pd.DataFrame()
        
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Could not get database connection")
                return pd.DataFrame()
            
            cur = conn.cursor()
            
            # Create placeholders for the IN clause
            counter_placeholders = ','.join(['%s' for _ in global_counters])
            
            query = f"""
                SELECT timestamp, camera_id, track_id, win_idx, frame_rel_idx, 
                       keypoints_json, pose_score
                FROM pose_keypoints 
                WHERE camera_id = %s 
                  AND track_id = %s 
                  AND win_idx IN ({counter_placeholders})
                ORDER BY win_idx, frame_rel_idx
            """
            
            # Build parameters list
            params = [camera_id, track_id] + global_counters
            
            cur.execute(query, params)
            results = cur.fetchall()
            
            # Convert to DataFrame
            columns = ['timestamp', 'camera_id', 'track_id', 'win_idx', 'frame_rel_idx', 
                      'keypoints_json', 'pose_score']
            df = pd.DataFrame(results, columns=columns)
            
            cur.close()
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching keypoint data: {e}")
            return pd.DataFrame()
    
    def process_matched_data_individually(self, matched_data: pd.DataFrame):
        """Process matched data grouped by exercise_log_id and call analyse_pose() for each exercise individually"""
        try:
            # Group by exercise_log_id to process each exercise log separately
            for exercise_log_id, exercise_data in matched_data.groupby('exercise_log_id'):
                try:
                    self.call_analyse_pose_for_exercise(exercise_data)
                    
                    with self._lock:
                        self.stats['total_analyse_calls'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing exercise log {exercise_log_id}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error processing matched data individually: {e}")
    
    def call_analyse_pose_for_exercise(self, exercise_data: pd.DataFrame):
        """Call analyse_pose() function with data for a single exercise"""
        try:
            # Extract exercise information
            first_row = exercise_data.iloc[0]
            exercise_name = first_row['exercise']
            reps = first_row['reps'] if pd.notna(first_row['reps']) else 0
            
            # Extract keypoints data
            keypoints_data = []
            for _, row in exercise_data.iterrows():
                try:
                    # Parse keypoints JSON
                    if isinstance(row['keypoints_json'], str):
                        kp_json = json.loads(row['keypoints_json'])
                    else:
                        kp_json = row['keypoints_json']
                    
                    keypoint_entry = {
                        'win_idx': row['win_idx'],
                        'frame_rel_idx': row['frame_rel_idx'],
                        'keypoints': kp_json.get('kps', []),
                        'pose_score': row['pose_score']
                    }
                    keypoints_data.append(keypoint_entry)
                    
                except Exception as e:
                    logger.warning(f"Error parsing keypoints for frame {row['frame_rel_idx']}: {e}")
                    continue
            
            if not keypoints_data:
                logger.warning(f"No valid keypoints data for exercise log {first_row['exercise_log_id']}")
                return
            
            # Additional metadata
            metadata = {
                'camera_id': first_row['camera_id'],
                'track_id': first_row['track_id'],
                'exercise_log_id': first_row['exercise_log_id'],
                'exercise_conf': first_row['exercise_conf'],
                'rep_conf': first_row['rep_conf'],
                'timestamp_exercise': first_row['timestamp_exercise'],
                'total_frames': len(keypoints_data)
            }
            
            # Call the analyse_pose function
            logger.info(f"Calling analyse_pose for exercise log {first_row['exercise_log_id']}: "
                    f"{exercise_name} ({reps} reps, {len(keypoints_data)} frames)")
            
            self.analyse_pose(
                keypoints=keypoints_data,
                exercise_name=exercise_name,
                reps=reps,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error calling analyse_pose: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        with self._lock:
            return self.stats.copy()


# Integration function for main.py
def integrate_keypoint_processor(db_manager, analyse_pose_func: Callable, 
                                processing_interval: int = 8) -> ContinuousKeypointProcessor:
    """
    Create and start the continuous keypoint processor
    
    Args:
        db_manager: DatabaseManager instance
        analyse_pose_func: Function to call with matched data
        processing_interval: How often to check for new data (seconds)
        
    Returns:
        ContinuousKeypointProcessor instance
    """
    processor = ContinuousKeypointProcessor(
        db_manager=db_manager,
        analyse_pose_func=analyse_pose_func,
        processing_interval=processing_interval
    )
    
    processor.start_continuous_processing()
    return processor


# Example analyse_pose function signature
def example_analyse_pose(keypoints: List[Dict], exercise_name: str, reps: int, metadata: Dict):
    """
    Example function signature for analyse_pose
    
    Args:
        keypoints: List of keypoint data for each frame
        exercise_name: Name of the exercise (e.g., "FrontRaises")
        reps: Number of reps detected by RepNet
        metadata: Additional metadata about the exercise
    """
    print(f"Analyzing {exercise_name} with {reps} reps ({len(keypoints)} frames)")
    print(f"Metadata: {metadata}")
    
    # Process keypoints data
    for i, kp_data in enumerate(keypoints):
        print(f"  Frame {i}: {len(kp_data['keypoints'])} keypoints, score: {kp_data['pose_score']:.3f}")