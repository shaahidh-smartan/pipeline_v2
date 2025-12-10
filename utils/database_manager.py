import psycopg2
from pgvector.psycopg2 import register_vector
import time
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv


class DatabaseManager:
    """Centralized database operations."""
    
    def __init__(self):
        # Load environment variables from .env.runtime file
        here = os.path.dirname(os.path.abspath(__file__))
        # adjust the .. to reach the repo root where .env.runtime actually is
        env_path = os.path.abspath(os.path.join(here, '..', '.env.runtime'))
        load_dotenv(env_path)
        print("[DB_DEBUG] PGPASSWORD present:", bool(os.getenv("PGPASSWORD")))

        
        self.db_config = {
            'dbname': os.getenv('PGDATABASE', 'person_identification_db'),
            'user': os.getenv('PGUSER', 'smartan'),
            'password': os.getenv('PGPASSWORD'),
            'host': os.getenv('PGHOST', 'localhost'),
            'port': os.getenv('PGPORT', '5432')
        }
    
    
    def get_connection(self):
        """Get database connection with pgvector support."""
        try:
            conn = psycopg2.connect(**self.db_config)
            register_vector(conn)
            return conn
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            return None
    
    # Add these methods to your DatabaseManager class

    def insert_pose_batch(self, pose_batch_data):
        """Insert pose batch data."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Parse timestamp if it's a string
            timestamp = pose_batch_data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            cur.execute("""
                INSERT INTO pose_batches (
                    timestamp, camera_id, track_id, win_idx, seq_start, seq_end,
                    avg_pose_score, total_frames
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (camera_id, track_id, win_idx) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    seq_start = EXCLUDED.seq_start,
                    seq_end = EXCLUDED.seq_end,
                    avg_pose_score = EXCLUDED.avg_pose_score,
                    total_frames = EXCLUDED.total_frames
            """, (
                timestamp,
                pose_batch_data['camera_id'],
                pose_batch_data['track_id'],
                pose_batch_data['win_idx'],
                pose_batch_data['seq_start'],
                pose_batch_data['seq_end'],
                pose_batch_data['avg_pose_score'],
                pose_batch_data['total_frames']
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to insert pose batch: {e}")
            return False

    def insert_pose_keypoints(self, keypoints_data):
        """Insert pose keypoints data."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Parse timestamp if it's a string
            timestamp = keypoints_data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Parse keypoints JSON if it's a string
            keypoints_json = keypoints_data['keypoints_json']
            if isinstance(keypoints_json, str):
                keypoints_json = json.loads(keypoints_json)
            
            cur.execute("""
                INSERT INTO pose_keypoints (
                    timestamp, camera_id, track_id, win_idx, frame_rel_idx,
                    keypoints_json, pose_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (camera_id, track_id, win_idx, frame_rel_idx) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    keypoints_json = EXCLUDED.keypoints_json,
                    pose_score = EXCLUDED.pose_score
            """, (
                timestamp,
                keypoints_data['camera_id'],
                keypoints_data['track_id'],
                keypoints_data['win_idx'],
                keypoints_data['frame_rel_idx'],
                json.dumps(keypoints_json),
                keypoints_data['pose_score']
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to insert pose keypoints: {e}")
            return False

    def batch_insert_pose_keypoints(self, keypoints_list):
        """Batch insert pose keypoints data."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_data = []
            for keypoints_data in keypoints_list:
                # Parse timestamp
                timestamp = keypoints_data['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Parse keypoints JSON
                keypoints_json = keypoints_data['keypoints_json']
                if isinstance(keypoints_json, str):
                    keypoints_json = json.loads(keypoints_json)
                
                insert_data.append((
                    timestamp,
                    keypoints_data['camera_id'],
                    keypoints_data['track_id'],
                    keypoints_data['win_idx'],
                    keypoints_data['frame_rel_idx'],
                    json.dumps(keypoints_json),
                    keypoints_data['pose_score']
                ))
            
            cur.executemany("""
                INSERT INTO pose_keypoints (
                    timestamp, camera_id, track_id, win_idx, frame_rel_idx,
                    keypoints_json, pose_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (camera_id, track_id, win_idx, frame_rel_idx) DO NOTHING
            """, insert_data)
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"[DB] Batch inserted {len(insert_data)} pose keypoints")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to batch insert pose keypoints: {e}")
            return False

    def insert_weight_detection(self, weight_data):
        """Insert weight detection data."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Parse timestamp if it's a string
            timestamp = weight_data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            cur.execute("""
                INSERT INTO weight_detections (
                    timestamp, camera_id, track_id, weight_label, confidence, bbox
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                timestamp,
                weight_data['camera_id'],
                weight_data['track_id'],
                weight_data['weight_label'],
                weight_data['confidence'],
                weight_data['bbox']
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to insert weight detection: {e}")
            return False

    # Update the existing insert_exercise_log method to handle global_counters
    def insert_exercise_log(self, exercise_data):
        """
        Insert exercise log entry into database with global_counters support.
        """
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Parse timestamp if it's a string
            timestamp = exercise_data['timestamp']
            if isinstance(timestamp, str):
                # Handle the %f literal in timestamp
                if '.%f' in timestamp:
                    timestamp = timestamp.replace('.%f', '.000000')
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            
            # Convert data to JSON - FIXED to handle global_counters properly
            vote_counts_json = json.dumps(exercise_data['vote_counts'])
            batch_ids_json = json.dumps(exercise_data['batch_ids'])
            
            # Fix: Ensure global_counters is always a list and properly converted to JSON
            global_counters = exercise_data.get('global_counters', [])
            if global_counters is None:
                global_counters = []
            global_counters_json = json.dumps(global_counters)
            
            # Debug logging to see what we're inserting
            print(f"[DB_DEBUG] Inserting exercise log for {exercise_data['camera_id']}:{exercise_data['track_id']}")
            print(f"[DB_DEBUG] global_counters raw: {global_counters}")
            print(f"[DB_DEBUG] global_counters_json: {global_counters_json}")
            
            cur.execute("""
                INSERT INTO exercise_logs (
                    timestamp, camera_id, track_id, exercise, exercise_conf,
                    reps, rep_conf, frame_count, voting_cycle_id, vote_counts,
                    batches_used, batch_ids, global_counters, entry_index, weight, weight_conf
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                timestamp,
                exercise_data['camera_id'],
                exercise_data['track_id'],
                exercise_data['exercise'],
                exercise_data['exercise_conf'],
                exercise_data['reps'],
                exercise_data['rep_conf'],
                exercise_data['frame_count'],
                exercise_data['voting_cycle_id'],
                vote_counts_json,
                exercise_data['batches_used'],
                batch_ids_json,
                global_counters_json,  # Fixed: Now properly converted to JSON
                exercise_data['entry_index'],
                exercise_data.get('weight', 'unknown'),
                exercise_data.get('weight_conf', 0.0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"[DB] Inserted exercise log: {exercise_data['camera_id']}:{exercise_data['track_id']} "
                f"- {exercise_data['exercise']} (reps: {exercise_data['reps']}) "
                f"global_counters: {global_counters}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to insert exercise log: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_pose_stats(self):
        """Get pose statistics from database."""
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            cur = conn.cursor()
            
            # Total pose batches
            cur.execute("SELECT COUNT(*) FROM pose_batches")
            total_batches = cur.fetchone()[0]
            
            # Total keypoints
            cur.execute("SELECT COUNT(*) FROM pose_keypoints")
            total_keypoints = cur.fetchone()[0]
            
            # Unique tracks
            cur.execute("SELECT COUNT(DISTINCT camera_id || ':' || track_id) FROM pose_batches")
            unique_tracks = cur.fetchone()[0]
            
            # Average pose scores
            cur.execute("SELECT AVG(avg_pose_score) FROM pose_batches")
            avg_pose_score = cur.fetchone()[0]
            
            # Recent activity (last 24 hours)
            cur.execute("""
                SELECT COUNT(*) FROM pose_batches 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            recent_batches = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return {
                'total_pose_batches': total_batches,
                'total_keypoints': total_keypoints,
                'unique_tracks': unique_tracks,
                'avg_pose_score': float(avg_pose_score) if avg_pose_score else 0.0,
                'recent_batches_24h': recent_batches
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get pose stats: {e}")
            return {}

    def get_weight_detection_stats(self):
        """Get weight detection statistics."""
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            cur = conn.cursor()
            
            # Total detections
            cur.execute("SELECT COUNT(*) FROM weight_detections")
            total_detections = cur.fetchone()[0]
            
            # Weight label counts
            cur.execute("""
                SELECT weight_label, COUNT(*) 
                FROM weight_detections 
                GROUP BY weight_label 
                ORDER BY COUNT(*) DESC
            """)
            weight_counts = dict(cur.fetchall())
            
            # Recent detections (last 24 hours)
            cur.execute("""
                SELECT COUNT(*) FROM weight_detections 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            recent_detections = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return {
                'total_weight_detections': total_detections,
                'weight_label_counts': weight_counts,
                'recent_detections_24h': recent_detections
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get weight detection stats: {e}")
            return {}

    def insert_face_embedding(self, person_name, embedding_vector, created_at):
        """Insert face embedding into database."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            timestamp = int(time.time()) if created_at is None else created_at
            
            cur.execute("""
                INSERT INTO face_embeddings (person_name, embedding, created_at)
                VALUES (%s, %s, %s)
            """, (person_name, embedding_vector.tolist(), timestamp))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to insert face embedding: {e}")
            return False

    def insert_person_reid_log(self, person_name, camera_id, track_id):
        """Insert person re-identification log into database."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO person_reid_mapped (person_name, camera_id, track_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (person_name, camera_id, track_id, detection_timestamp) DO NOTHING
            """, (person_name, camera_id, track_id))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to insert person reid log: {e}")
            return False


    def get_exercise_stats(self):
        """Get exercise statistics from database."""
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            cur = conn.cursor()
            
            # Total logs
            cur.execute("SELECT COUNT(*) FROM exercise_logs")
            total_logs = cur.fetchone()[0]
            
            # Unique tracks
            cur.execute("SELECT COUNT(DISTINCT camera_id || ':' || track_id) FROM exercise_logs")
            unique_tracks = cur.fetchone()[0]
            
            # Exercise types
            cur.execute("SELECT exercise, COUNT(*) FROM exercise_logs GROUP BY exercise ORDER BY COUNT(*) DESC")
            exercise_counts = dict(cur.fetchall())
            
            # Average reps per exercise
            cur.execute("""
                SELECT exercise, AVG(reps) 
                FROM exercise_logs 
                WHERE reps IS NOT NULL 
                GROUP BY exercise 
                ORDER BY AVG(reps) DESC
            """)
            avg_reps = dict(cur.fetchall())
            
            # Recent activity (last 24 hours)
            cur.execute("""
                SELECT COUNT(*) 
                FROM exercise_logs 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            recent_logs = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return {
                'total_exercise_logs': total_logs,
                'unique_tracks': unique_tracks,
                'exercise_counts': exercise_counts,
                'avg_reps_per_exercise': avg_reps,
                'recent_logs_24h': recent_logs
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get exercise stats: {e}")
            return {}

    def get_track_exercise_history(self, camera_id, track_id, limit=50):
        """Get exercise history for a specific track."""
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT timestamp, exercise, exercise_conf, reps, rep_conf, 
                       voting_cycle_id, vote_counts, entry_index
                FROM exercise_logs 
                WHERE camera_id = %s AND track_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (camera_id, track_id, limit))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return [{
                'timestamp': row[0],
                'exercise': row[1],
                'exercise_conf': float(row[2]),
                'reps': row[3],
                'rep_conf': float(row[4]) if row[4] else None,
                'voting_cycle_id': row[5],
                'vote_counts': json.loads(row[6]) if row[6] else {},
                'entry_index': row[7]
            } for row in results]
            
        except Exception as e:
            print(f"[ERROR] Failed to get track history: {e}")
            return []

    def insert_person_embedding(self, person_name, embedding, camera_id):
        """Insert person body embedding into database."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Ensure embedding is normalized and correct type
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)
                embedding_list = embedding.tolist()
            else:
                embedding_list = embedding
            
            cur.execute("""
                INSERT INTO vgg_embeddings (person_name, embedding, camera_id, count)
                VALUES (%s, %s, %s, %s)
            """, (person_name, embedding_list, camera_id, 1))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to insert person embedding: {e}")
            return False
    
    def insert_bpbreid_embedding(self, face_id, embedding, visibility, camera_id):
        """
        Insert BPBreID person re-identification embedding into database.
        Stores each body part as a SEPARATE row with the same face_id (PID).

        Args:
            face_id: Foreign key to face_embeddings table (used as PID)
            embedding: torch.Tensor [6, 512] or numpy array - 6 body parts
            visibility: torch.Tensor [6] or numpy array - visibility scores
            camera_id: Camera identifier

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cur = conn.cursor()

            # Convert to numpy if torch tensor
            if hasattr(embedding, 'cpu'):  # torch.Tensor
                embedding = embedding.cpu().numpy()
            if hasattr(visibility, 'cpu'):  # torch.Tensor
                visibility = visibility.cpu().numpy()

            # Insert each body part as a separate row
            # embedding shape: [6, 512], visibility shape: [6]
            for body_part_id in range(6):
                part_embedding = embedding[body_part_id].astype(np.float32).tolist()  # [512]
                part_visibility = float(visibility[body_part_id])  # scalar

                cur.execute("""
                    INSERT INTO bpbreid_embeddings
                    (face_id, body_part_id, embedding, visibility, camera_id)
                    VALUES (%s, %s, %s, %s, %s)
                """, (face_id, body_part_id, part_embedding, part_visibility, camera_id))

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to insert BPBreID embedding: {e}")
            return False

    def get_person_embedding_count(self, person_name):
        """Get count of embeddings for a person."""
        try:
            conn = self.get_connection()
            if not conn:
                return 0

            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM vgg_embeddings WHERE person_name = %s", (person_name,))
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            print(f"[ERROR] Failed to get embedding count: {e}")
            return 0

    def get_bpbreid_embedding_count(self, face_id):
        """
        Get count of BPBreID frame embeddings for a person by face_id.
        Since each frame has 6 body parts stored separately, we count distinct frames.
        """
        try:
            conn = self.get_connection()
            if not conn:
                return 0

            cur = conn.cursor()

            # Count distinct frames (group by camera_id and created_at)
            # Each frame produces 6 rows (one per body part)
            cur.execute("""
                SELECT COUNT(DISTINCT (camera_id, created_at))
                FROM bpbreid_embeddings
                WHERE face_id = %s
            """, (face_id,))

            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            print(f"[ERROR] Failed to get BPBreID embedding count: {e}")
            return 0
    
    def clear_person_embeddings(self):
        """Clear all person embeddings."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            cur.execute("DELETE FROM vgg_embeddings")
            deleted_count = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"[DB] Cleared {deleted_count} person embeddings")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to clear embeddings: {e}")
            return False
    
    def test_connection(self):
        """Test database connection and create tables if needed."""
        try:
            conn = self.get_connection()
            if not conn:
                return False, "Could not establish connection"
            
            cur = conn.cursor()
            
            # Test basic connectivity
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"[DB_TEST] Connected to PostgreSQL: {version[0] if version else 'Unknown version'}")
            
            # Check if required tables exist
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('face_embeddings', 'vgg_embeddings', 'person_reid_mapped')
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            print(f"[DB_TEST] Existing tables: {existing_tables}")
            
            # Check if exercise_logs table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'exercise_logs'
                )
            """)
            exercise_table_exists = cur.fetchone()[0]
            
            if not exercise_table_exists:
                print("[DB_TEST] exercise_logs table doesn't exist - will be created when needed")
            else:
                print("[DB_TEST] exercise_logs table exists")
            
            cur.close()
            conn.close()
            return True, f"Connected successfully. Tables found: {existing_tables}"
            
        except Exception as e:
            return False, f"Connection test failed: {e}"

    def create_exercise_logs_table(self):
        """Create the exercise_logs table if it doesn't exist."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Create the exercise_logs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exercise_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    camera_id VARCHAR(50) NOT NULL,
                    track_id INTEGER NOT NULL,
                    exercise VARCHAR(100) NOT NULL,
                    exercise_conf DECIMAL(5,3) NOT NULL,
                    reps INTEGER,
                    rep_conf DECIMAL(5,3),
                    frame_count INTEGER NOT NULL,
                    voting_cycle_id VARCHAR(100) NOT NULL,
                    vote_counts JSONB NOT NULL,
                    batches_used INTEGER NOT NULL,
                    batch_ids JSONB NOT NULL,
                    entry_index INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_logs_camera_track 
                ON exercise_logs(camera_id, track_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_logs_exercise 
                ON exercise_logs(exercise)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_logs_timestamp 
                ON exercise_logs(timestamp)
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
            print("[DB_CREATE] exercise_logs table and indexes created successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create exercise_logs table: {e}")
            return False
        
    def get_database_stats(self):
        """Get database statistics."""
        try:
            conn = self.get_connection()
            if not conn:
                return {'total_people': 0, 'total_embeddings': 0}
            
            cur = conn.cursor()
            
            # Face embeddings stats
            cur.execute("SELECT COUNT(DISTINCT person_name) FROM face_embeddings")
            face_people = cur.fetchone()[0]
            
            # Person embeddings stats  
            cur.execute("SELECT COUNT(DISTINCT person_name) as people, COUNT(*) as total_embeddings FROM vgg_embeddings")
            person_people, person_embeddings = cur.fetchone()
            
            cur.close()
            conn.close()
            
            return {
                'face_people': face_people,
                'person_people': person_people,
                'person_embeddings': person_embeddings
            }
        except Exception as e:
            print(f"[ERROR] Failed to get database stats: {e}")
            return {'face_people': 0, 'person_people': 0, 'person_embeddings': 0}
