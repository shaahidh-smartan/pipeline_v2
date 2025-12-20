import pandas as pd
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import threading
import time
from psycopg2.extras import Json
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Import ALL 15 exercise analysis modules
from modules.bicep_curl import BicepCurls
from modules.db_chest_press import DB_Chest_Press
from modules.db_incline_chest_press import DB_Incline_Chest_Press
from modules.db_lunges import DB_Lunges
from modules.db_reverse_flys import DB_Reverse_Flys
from modules.front_raise import FrontRaise
from modules.hammer_curls import Hammer_Curls
from modules.kb_goblet_squats import KB_Goblet_Squats
from modules.kb_goodmorning import KB_Goodmorning
from modules.kb_overhead_press import KB_Overhead_Press
from modules.kb_swings import KB_Swings
from modules.lateral_raise import Lateral_Raise
from modules.seated_db_shoulder_press import Seated_DB_Shoulder_Press
from modules.singlearm_db_rows import Singlearm_DB_Rows
from modules.upright_rows import Upright_Rows

logger = logging.getLogger(__name__)

class RealTimePoseAnalyzer:
    """Real-time pose analysis integration with exercise-specific modules"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.init_exercise_mapping()
        self.init_database_schema()
        
        # Statistics
        self.analysis_stats = {
            'total_analyzed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'unknown_exercises': 0,
            'database_updates': 0,
            'analysis_times': []
        }
        
    def init_exercise_mapping(self):
        """Initialize exercise name mapping from pipeline to analyzer classes"""
        self.EXERCISE_NAME_MAP = {
            # Pipeline exercise names -> Analyzer class names
            'Bicep Curl': 'BicepCurls',
            'Front Raise': 'FrontRaises', 
            'Hammer Curl': 'Hammer_Curls',
            'Lateral Raise': 'Lateral_Raise',
            'Upright Rows': 'Upright_Rows',
            'DB Chest Press': 'DB_Chest_Press',
            'DB Incline Chest Press': 'DB_Incline_Chest_Press', 
            'DB Lunges': 'DB_Lunges',
            'DB Reverse Flys': 'DB_Reverse_Flys',
            'KB Goblet Squat': 'KB_Goblet_Squats',
            'KB Overhead Press': 'KB_Overhead_Press',
            'KB Swings': 'KB_Swings',
            'KB Goodmorning': 'KB_Goodmorning',
            'Seated DB Shoulder Pess': 'Seated_DB_Shoulder_Press',  # Note: typo in original
            'Single Arm DB Row': 'Singlearm_DB_Rows'
        }
        
        self.ANALYZER_CLASSES = {
            'BicepCurls': BicepCurls,
            'FrontRaises': FrontRaise,
            'Hammer_Curls': Hammer_Curls,
            'Lateral_Raise': Lateral_Raise,
            'Upright_Rows': Upright_Rows,
            'DB_Chest_Press': DB_Chest_Press,
            'DB_Incline_Chest_Press': DB_Incline_Chest_Press,
            'DB_Lunges': DB_Lunges,
            'DB_Reverse_Flys': DB_Reverse_Flys,
            'KB_Goblet_Squats': KB_Goblet_Squats,
            'KB_Overhead_Press': KB_Overhead_Press,
            'KB_Swings': KB_Swings,
            'KB_Goodmorning': KB_Goodmorning,
            'Seated_DB_Shoulder_Press': Seated_DB_Shoulder_Press,
            'Singlearm_DB_Rows': Singlearm_DB_Rows
        }


    @staticmethod
    def json_safe(obj):
        import numpy as np, math
        from datetime import datetime, date
        try:
            import pandas as pd
            _has_pd = True
        except Exception:
            _has_pd = False

        if obj is None or isinstance(obj, (bool, str, int, float)):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(obj, (np.bool_,)):     return bool(obj)
        if isinstance(obj, (np.ndarray,)):   return [RealTimePoseAnalyzer.json_safe(x) for x in obj.tolist()]
        if isinstance(obj, (datetime, date)): return obj.isoformat()
        if _has_pd and isinstance(obj, getattr(pd, "Timestamp")): return obj.isoformat()
        if isinstance(obj, (np.datetime64,)): return str(obj.astype('datetime64[ns]'))
        if isinstance(obj, dict):            return {str(k): RealTimePoseAnalyzer.json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)): return [RealTimePoseAnalyzer.json_safe(x) for x in obj]
        return str(obj)

        
    def init_database_schema(self):
        """Add pose_feedback column to exercise_logs table if it doesn't exist"""
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Could not get database connection for schema init")
                return
                
            cur = conn.cursor()
            
            # Check if pose_feedback column exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema. # try:
            #     cleaned_result = convert_numpy_types(analysis_result)
            #     feedback_json = str(json.dumps(cleaned_result))
            # except Exception as json_error:
            #     logger.error(f"Error converting and serializing analysis result: {json_error}")
            #     return Falsecolumns 
                WHERE table_name='exercise_logs' AND column_name='pose_feedback'
            """)
            
            if not cur.fetchone():
                # Add the column
                cur.execute("""
                    ALTER TABLE exercise_logs 
                    ADD COLUMN pose_feedback JSONB DEFAULT NULL
                """)
                conn.commit()
                logger.info("Added pose_feedback column to exercise_logs table")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")



    # INSIDE RealTimePoseAnalyzer.update_exercise_log_with_feedback

    def update_exercise_log_with_feedback(self, exercise_log_id: int, analysis_result: Dict) -> bool:
        import json, psycopg2
        from psycopg2.extras import Json

        # 1) Normalize payload fully
        clean = self.json_safe(analysis_result)

        # 2) Normalize the ID (this is causing your current error)
        try:
            ex_id = int(exercise_log_id)   # <-- cast numpy.int64 -> int
        except Exception:
            # last resort: string (works too), but prefer int
            ex_id = str(exercise_log_id)

        # Optional: sanity checks before DB call
        # assert not any(isinstance(v, np.generic) for v in _walk_types(clean)), "NumPy in payload"

        sql = """
            UPDATE exercise_logs
            SET analysis_json = %s
            WHERE id = %s
        """

        dumps = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

        conn = self.db_manager.get_connection()
        if not conn:
            logger.error("DB feedback update: no connection")
            return False

        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (Json(clean, dumps=dumps), ex_id))
                    updated = cur.rowcount
            if updated > 0:
                self.analysis_stats['database_updates'] += 1
                logger.info("Updated exercise_logs.id=%s pose_f", ex_id)
                return True
            logger.warning("No rows updated for exercise_log_id=%s", ex_id)
            return False
        except psycopg2.Error as db_err:
            logger.exception("DB error updating pose_f for id=%s: %s", ex_id, db_err)
            return False




    def analyze_exercise(self, keypoints: List[Dict], exercise_name: str, reps: int, metadata: Dict) -> Optional[Dict]:
        """
        Analyze exercise pose using appropriate module
        
        Args:
            keypoints: List of keypoint data for each frame
            exercise_name: Exercise type (e.g., "Bicep Curl")
            reps: Number of reps from RepNet
            metadata: Additional info (camera_id, track_id, exercise_log_id, etc.)
            
        Returns:
            Analysis results dictionary or None if failed
        """
        start_time = time.time()
        
        try:
            self.analysis_stats['total_analyzed'] += 1
            
            # Map exercise name to analyzer class
            analyzer_class_name = self.EXERCISE_NAME_MAP.get(exercise_name)
            if not analyzer_class_name:
                logger.warning(f"Unknown exercise type: {exercise_name}")
                self.analysis_stats['unknown_exercises'] += 1
                return None
                
            analyzer_class = self.ANALYZER_CLASSES.get(analyzer_class_name)
            if not analyzer_class:
                logger.error(f"Analyzer class not found: {analyzer_class_name}")
                self.analysis_stats['failed_analyses'] += 1
                return None
            
            # Transform keypoints data to DataFrame format expected by analyzers
            df = self.create_analysis_dataframe(keypoints, exercise_name, reps, metadata)
            
            if df.empty:
                logger.warning(f"Empty dataframe for {exercise_name} analysis")
                self.analysis_stats['failed_analyses'] += 1
                return None
            
            # Initialize analyzer and run analysis
            analyzer = analyzer_class()
            logger.info(f"Running {analyzer_class_name} analysis on {len(keypoints)} frames")
            
            # Run the analysis
            analysis_result = analyzer.analyze_batch(df.copy())
            
            # Add metadata to results
            analysis_result.update({
                'exercise_name': exercise_name,
                'analyzer_used': analyzer_class_name,
                'total_frames_analyzed': len(keypoints),
                'analysis_timestamp': time.time(),
                'metadata': {
                    'camera_id': metadata.get('camera_id'),
                    'track_id': metadata.get('track_id'),
                    'exercise_log_id': metadata.get('exercise_log_id'),
                    'input_reps': reps
                }
            })
            
            analysis_time = time.time() - start_time
            self.analysis_stats['analysis_times'].append(analysis_time)
            self.analysis_stats['successful_analyses'] += 1
            
            logger.info(f"Successfully analyzed {exercise_name} in {analysis_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            self.analysis_stats['failed_analyses'] += 1
            logger.error(f"Error analyzing {exercise_name}: {e}")
            return None
    
    def create_analysis_dataframe(self, keypoints: List[Dict], exercise_name: str, reps: int, metadata: Dict) -> pd.DataFrame:
        """Transform keypoints data to DataFrame format expected by analysis modules"""
        try:
            if not keypoints:
                return pd.DataFrame()
            
            # Build DataFrame rows
            rows = []
            for kp_data in keypoints:
                # Convert keypoints to the JSON format expected by analyzers
                keypoints_json = json.dumps({"kps": kp_data['keypoints']})
                
                row = {
                    'track_id': metadata.get('track_id', 0),
                    'win_idx': kp_data.get('win_idx', 0),
                    'frame_rel_idx': kp_data.get('frame_rel_idx', 0),
                    'exercise': exercise_name,
                    'rep': reps if reps is not None else 0,
                    'keypoints_json': keypoints_json,
                    'pose_score': kp_data.get('pose_score', 0.0)
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            logger.debug(f"Created DataFrame with {len(df)} rows for {exercise_name} analysis")
            return df
            
        except Exception as e:
            logger.error(f"Error creating analysis DataFrame: {e}")
            return pd.DataFrame()
    
    # def update_exercise_log_with_feedback(self, exercise_log_id: int, analysis_result: Dict) -> bool:
    #     """Update exercise_logs table with pose analysis feedback"""
    #     try:
    #         conn = self.db_manager.get_connection()
    #         if not conn:
    #             logger.error("Could not get database connection for feedback update")
    #             return False
            
    #         cur = conn.cursor()

    #         print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #         print(analysis_result)
            
    #         # Convert all NumPy types to Python native types
    #         # try:
    #         #     cleaned_result = convert_numpy_types(analysis_result)
    #         #     feedback_json = str(json.dumps(cleaned_result))
    #         # except Exception as json_error:
    #         #     logger.error(f"Error converting and serializing analysis result: {json_error}")
    #         #     return False
            
    #         # Update the exercise log with pose feedback
    #         update_query = """
    #             UPDATE exercise_logs 
    #             SET pose_f = %s
    #             WHERE id = %s
    #         """
    #         cur.execute(
    #             update_query,[analysis_result, exercise_log_id]
    #         )

    #         # cur.execute(update_query, [feedback_json, exercise_log_id])
    #         conn.commit()
            
    #         if cur.rowcount > 0:
    #             self.analysis_stats['database_updates'] += 1
    #             logger.info(f"Updated exercise log {exercise_log_id} with pose feedback")
    #             result = True
    #         else:
    #             logger.warning(f"No rows updated for exercise_log_id {exercise_log_id}")
    #             result = False
            
    #         cur.close()
    #         conn.close()
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"Error updating exercise log with feedback: {e}")
    #         return False
    





    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        stats = self.analysis_stats.copy()
        
        # Calculate average analysis time
        if stats['analysis_times']:
            stats['avg_analysis_time'] = sum(stats['analysis_times']) / len(stats['analysis_times'])
            stats['max_analysis_time'] = max(stats['analysis_times'])
            stats['min_analysis_time'] = min(stats['analysis_times'])
        else:
            stats['avg_analysis_time'] = 0
            stats['max_analysis_time'] = 0
            stats['min_analysis_time'] = 0
        
        # Calculate success rate
        if stats['total_analyzed'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_analyzed'] * 100
        else:
            stats['success_rate'] = 0
            
        return stats









# Global pose analyzer instance
pose_analyzer = None

def init_pose_analyzer(db_manager):
    """Initialize the global pose analyzer"""
    global pose_analyzer
    pose_analyzer = RealTimePoseAnalyzer(db_manager)
    logger.info("Real-time pose analyzer initialized")
    return pose_analyzer

def analyse_pose(keypoints: List[Dict], exercise_name: str, reps: int, metadata: Dict):
    """
    Enhanced analyze pose function that performs real pose analysis
    
    Args:
        keypoints: List of keypoint data for each frame
        exercise_name: Exercise type (e.g., "Bicep Curl")
        reps: Number of reps from RepNet
        metadata: Additional info (camera_id, track_id, exercise_log_id, etc.)
    """
    global pose_analyzer
    
    if pose_analyzer is None:
        logger.error("Pose analyzer not initialized")
        return
    
    exercise_log_id = metadata.get('exercise_log_id')
    
    logger.info(f"[POSE_ANALYSIS] Starting analysis for {exercise_name}: {reps} reps, {len(keypoints)} frames")
    logger.info(f"[POSE_ANALYSIS] Exercise Log ID: {exercise_log_id}")
    logger.info(f"[POSE_ANALYSIS] Camera: {metadata['camera_id']}, Track: {metadata['track_id']}")
    
    # Run the pose analysis in a separate thread to avoid blocking
    def analyze_async():
        print("ASync >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        try:
            # Perform the analysis
            analysis_result = pose_analyzer.analyze_exercise(keypoints, exercise_name, reps, metadata)
           
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(analysis_result)
            
            if analysis_result is None:
                logger.error(f"[POSE_ANALYSIS] Analysis failed for exercise log {exercise_log_id}")
                return
            
            # Update the database with results
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(exercise_log_id)
            if exercise_log_id:
                
                success = pose_analyzer.update_exercise_log_with_feedback(exercise_log_id, analysis_result)
                if success:
                    logger.info(f"[POSE_ANALYSIS] Successfully updated exercise log {exercise_log_id} with feedback")
                else:
                    logger.error(f"[POSE_ANALYSIS] Failed to update exercise log {exercise_log_id}")
            
            logger.info(f"[POSE_ANALYSIS] Completed analysis for exercise log {exercise_log_id}")
            
        except Exception as e:
            logger.error(f"[POSE_ANALYSIS] Error in async analysis: {e}")
    
    # Run analysis in background thread
    analysis_thread = threading.Thread(target=analyze_async, name=f"PoseAnalysis-{exercise_log_id}")
    analysis_thread.daemon = True
    analysis_thread.start()

def get_pose_analysis_stats():
    """Get pose analysis statistics"""
    global pose_analyzer
    if pose_analyzer is None:
        return {"error": "Pose analyzer not initialized"}
    return pose_analyzer.get_analysis_stats()