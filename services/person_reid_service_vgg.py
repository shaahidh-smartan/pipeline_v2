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

# ByteTracker imports
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

# Import camera management
from camera import CameraManager, create_camera_configs_from_ips

# Import our utility modules
from utils.database_manager import DatabaseManager
from utils.person_embedder import PersonEmbedder
from utils.person_detector import PersonDetector
from utils.similarity_search import SimilaritySearch


class PersonReIDService:
    """
    Person Re-Identification Service.
    
    This service:
    1. Uses collected body embeddings from the database
    2. Tracks persons across multiple cameras
    3. Identifies persons using similarity search against stored embeddings
    4. Provides real-time person tracking and identification
    """
    
    def __init__(self, stream_configs, similarity_threshold=0.8, use_batch_yolo=True, visual_mode=True):
        """
        Initialize Dynamic Camera Person ReID Service.

        Args:
            stream_configs (list): List of camera stream configurations
            similarity_threshold (float): Threshold for person similarity matching (default: 0.8)
            use_batch_yolo (bool): Enable batch YOLO processing for 6-9x speedup (default: True)
            visual_mode (bool): Enable visual output (drawing and display). Set to False for performance testing (default: True)

        Raises:
            ValueError: If no cameras are provided
        """
        self.num_cameras = len(stream_configs)
        if self.num_cameras == 0:
            raise ValueError("At least one camera is required")

        self.stream_configs = stream_configs
        self.similarity_threshold = similarity_threshold
        self.use_batch_yolo = use_batch_yolo  # Batch processing flag
        self.visual_mode = visual_mode  # Visual output flag
        self.last_detection_time = {}
        self.reset_delay = 4  # seconds

        if self.use_batch_yolo:
            print(f"[INFO] Batch YOLO enabled for {self.num_cameras} cameras (expected 6-9x speedup)")
        else:
            print(f"[INFO] Using per-camera YOLO processing (legacy mode)")

        if not self.visual_mode:
            print(f"[INFO] Visual mode DISABLED - No drawing or display (performance testing mode)")
        else:
            print(f"[INFO] Visual mode ENABLED - Drawing and display active")

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
        
        print("Initializing person embedder...")
        self.person_embedder = PersonEmbedder()
        
        print("Initializing person detector...")
        self.person_detector = PersonDetector()
        
        print("Initializing similarity search...")
        self.similarity_search = SimilaritySearch(self.db_manager)
        
        # Add caching for tracking ID to person mapping
        self.track_id_to_person = {}  # {(camera_id, track_id): {'name': str, 'confidence': float, 'timestamp': float}}
        self.confidence_threshold_for_caching = 0.75  # Cache if confidence >= 0.8
        self.cache_expiry_time = 30.0  # Cache expires after 30 seconds of no updates
        
        # Create tracker arguments
        self.tracker_args = type('Args', (object,), {
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
            self.trackers[camera_id] = BYTETracker(self.tracker_args)
        
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
        self.last_frame_count = {}  # Track last processed frame count to avoid counting duplicates
        for i in range(self.num_cameras):
            camera_id = f"cam_{i+1}"
            self.fps_counters[camera_id] = 0
            self.fps_start_times[camera_id] = time.time()
            self.current_fps[camera_id] = 0
            self.last_frame_count[camera_id] = -1
        
        # Initialize database connection first
        self.initialize_database_connection()
        self.master_person_list = []

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
    
    def get_cached_person(self, camera_id, track_id):
        """
        Get cached person identification for a track ID.

        Retrieves cached person identification if available and not expired.
        Automatically removes expired cache entries.

        Args:
            camera_id (str): Camera identifier
            track_id (int): Track identifier

        Returns:
            dict or None: Cached person data with name, confidence, and timestamp, or None if not cached/expired
        """
        cache_key = (camera_id, track_id)
        
        if cache_key in self.track_id_to_person:
            cached_entry = self.track_id_to_person[cache_key]
            current_time = time.time()
            
            # Check if cache entry hasn't expired
            if current_time - cached_entry['timestamp'] <= self.cache_expiry_time:
                return cached_entry
            else:
                # Cache expired, remove it
                print(f"[CACHE EXPIRED] Removing expired entry for track ID {track_id}")
                del self.track_id_to_person[cache_key]
        
        return None

    def cache_person_identification(self, camera_id, track_id, person_name, confidence):
        """
        Cache person identification if confidence is high enough.

        Stores person identification in cache for fast retrieval if confidence
        meets the minimum threshold for caching.

        Args:
            camera_id (str): Camera identifier
            track_id (int): Track identifier
            person_name (str): Identified person name
            confidence (float): Confidence score of the identification

        Returns:
            bool: True if cached, False if confidence too low
        """
        if confidence >= self.confidence_threshold_for_caching:
            cache_key = (camera_id, track_id)
            self.track_id_to_person[cache_key] = {
                'name': person_name,
                'confidence': confidence,
                'timestamp': time.time()
            }
            print(f"[CACHE STORED] Track ID {track_id} -> {person_name} (confidence: {confidence:.3f})")
            return True
        return False

    def update_cache_timestamp(self, camera_id, track_id):
        """
        Update timestamp for cached entry to keep it alive.

        Extends the lifetime of a cached entry by updating its timestamp.

        Args:
            camera_id (str): Camera identifier
            track_id (int): Track identifier
        """
        cache_key = (camera_id, track_id)
        if cache_key in self.track_id_to_person:
            self.track_id_to_person[cache_key]['timestamp'] = time.time()


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
        Process frame for person re-identification with database similarity search.

        Main processing function that detects persons, tracks them, and performs
        person identification using database similarity search with caching.

        Args:
            frame (numpy.ndarray): Input video frame
            camera_id (str): Camera identifier

        Returns:
            list: List of person re-identification results with tracking and match information
        """
        try:
            # Detect persons using person detector utility
            person_detections, person_boxes_track = self.person_detector.detect_persons(frame)

            now = time.time()
            # Extract camera number from camera_id (e.g., 'cam_1' -> 1)
            sample_id = int(camera_id.split('_')[1]) - 1

            if len(person_boxes_track) > 0:
                self.last_detection_time[sample_id] = now
            else:
                last_seen = self.last_detection_time.get(sample_id, now)

                if now - last_seen > 0.1:
                    # print(f"[DEBUG] No person detected for 0.1s on camera {sample_id}, clearing tracking history")
                    
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
                        
                        # Clear cache for this camera when tracking is reset
                        keys_to_remove = [key for key in self.track_id_to_person.keys() if key[0] == camera_id]
                        for key in keys_to_remove:
                            del self.track_id_to_person[key]
     
                    except Exception as e:
                        print(f"Error updating tracker with empty values: {e}")
                    
                    self.last_detection_time[sample_id] = now

            if not person_detections or not person_boxes_track:
                return []

            target_box = []
            reid_results = []
            similarity_check = 0
            
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
            
            # Process each target for ReID with database similarity search and caching
            for target in target_box:
                track_id, x, y, w, h, _, cam_id = target
                
                # Check cache first
                cached_person = self.get_cached_person(cam_id, track_id)
                
                if cached_person:
                    # Use cached result
                    person_name = cached_person['name']
                    similarity = cached_person['confidence']
                    
                    # Update cache timestamp to keep it alive
                    self.update_cache_timestamp(cam_id, track_id)
                    
                    # Convert coordinates
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    bbox = [x1, y1, x2, y2]
                    self.cache_master_list(cached_person['name'], cam_id, track_id)
                    # Create match object for consistency
                    match = {
                        'person_name': person_name,
                        'similarity': similarity,
                        'source_camera': 'cached'
                    } if person_name != "Unknown" else None
                    
                    reid_results.append({
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'match': match,
                        'similarity': similarity,
                        'person_name': person_name,
                        'camera_id': cam_id,
                        'cached': True  # Flag to indicate this was from cache
                    })
                    
                else:
                    # No cache hit, perform database ReID
                    # Convert from TLWH to TLBR
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    
                    # Get person crop using person detector utility
                    person_crop = self.person_detector.get_person_crop(frame, [x1, y1, x2, y2], padding=10)
                    
                    if person_crop is None:
                        print(f"Empty crop for track_id {track_id}, skipping")
                        continue
                    
                    try:
                        # Extract embedding using person embedder
                        embedding = self.person_embedder.extract_embedding(person_crop)
                        
                        if embedding is not None:
                            # Database similarity search using similarity search utility
                            match, similarity = self.similarity_search.find_person_match_euclidean(
                                embedding, threshold=self.similarity_threshold
                            )
                            similarity_check += 1
                            
                            if match is not None:
                                person_name = match.get('person_name')
                                # Try to cache this result if confidence is high
                                self.cache_person_identification(cam_id, track_id, person_name, similarity)
                            else:
                                person_name = "Unknown"
                                # Cache "Unknown" only if we're very confident it's not a match
                                if similarity < 0.3:  # Very low similarity, likely not anyone we know
                                    self.cache_person_identification(cam_id, track_id, person_name, 1.0 - similarity)
                            
                            # Store detailed results
                            reid_results.append({
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2],
                                'match': match,
                                'similarity': similarity,
                                'person_name': person_name,
                                'camera_id': cam_id,
                                'cached': False  # Flag to indicate this was freshly computed
                            })
                        else:
                            print(f"Track ID {track_id}: Failed to extract embedding")
                            
                    except Exception as e:
                        print(f"Error processing track_id {track_id}: {e}")
                        continue
            # print("==========================End of reid========================")
            return reid_results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return []

    def cleanup_expired_cache_entries(self):
        """
        Periodically clean up expired cache entries.

        Removes cache entries that have exceeded the maximum age to prevent
        memory leaks and maintain cache freshness.
        """
        current_time = time.time()
        expired_keys = [
            key for key, value in self.track_id_to_person.items()
            if current_time - value['timestamp'] > self.cache_expiry_time
        ]
        
        for key in expired_keys:
            del self.track_id_to_person[key]
        
        if expired_keys:
            print(f"[CACHE CLEANUP] Removed {len(expired_keys)} expired entries")

    def get_cache_info(self):
        """
        Get information about the current cache state.

        Returns detailed information about all cached entries including
        track IDs, person names, confidence scores, and cache age.

        Returns:
            dict: Dictionary containing cache information organized by camera
        """
        cache_info = {}
        for (camera_id, track_id), data in self.track_id_to_person.items():
            if camera_id not in cache_info:
                cache_info[camera_id] = []
            cache_info[camera_id].append({
                'track_id': track_id,
                'person_name': data['name'],
                'confidence': data['confidence'],
                'age_seconds': time.time() - data['timestamp']
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

                record = camera.get_frame()

                if record is not None:
                    frame = record["frame"]
                    pts_ns = record.get("pts_ns", -1)
                    arrive_ns = record.get("arrive_ns", None)
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
                    if self.visual_mode:
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
                        # In non-visual mode, store original frame without drawing
                        camera.set_processed_frame(frame)

                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error in processing thread for {camera_id}: {e}")
                time.sleep(0.1)

    def _submit_crops_for_exercise(self, frame, reid_results, camera_id):
        """
        Hook method for submitting person crops for exercise detection.

        This is a no-op in the base class. Subclasses (like BridgeReIDService)
        can override this to submit crops to their exercise detection pipeline.

        Args:
            frame: The video frame
            reid_results: List of ReID results with bbox and track_id
            camera_id: Camera identifier
        """
        # Base class does nothing - subclasses override this
        pass

    def batched_processing_loop(self):
        """
        Main batch processing loop for all cameras.

        Collects frames from all cameras and processes them in a single batch,
        dramatically improving YOLO throughput (6-9x faster).

        This replaces the per-camera processing threads when use_batch_yolo=True.
        """
        print("Starting batched processing loop for all cameras")

        # Initialize persistent detection data for all cameras
        persistent_detections = {cam_id: [] for cam_id in self.camera_manager.get_camera_ids()}
        detection_timestamps = {cam_id: 0 for cam_id in self.camera_manager.get_camera_ids()}

        # Performance tracking
        batch_count = 0
        timing_sum = {'collect': 0, 'yolo': 0, 'total': 0}
        frame_count_sum = 0

        while self.running:
            try:
                batch_start = time.time()

                # Step 1: Collect frames from all cameras (with timeout)
                frames_dict, records_dict = self.collect_frames_for_batch(timeout_ms=10)
                collect_time = time.time() - batch_start

                if not frames_dict:
                    time.sleep(0.001)  # Brief pause if no frames ready
                    continue

                yolo_start = time.time()
                # Step 2: Batch YOLO + Tracking + ReID
                reid_results_dict = self.process_frames_batch_for_reid(frames_dict)
                yolo_time = time.time() - yolo_start
                total_time = time.time() - batch_start

                # Track timing
                batch_count += 1
                timing_sum['collect'] += collect_time
                timing_sum['yolo'] += yolo_time
                timing_sum['total'] += total_time
                frame_count_sum += len(frames_dict)

                # Print performance stats every 50 batches (~2-3 seconds)
                if batch_count % 50 == 0:
                    avg_collect = (timing_sum['collect'] / 50) * 1000
                    avg_yolo = (timing_sum['yolo'] / 50) * 1000
                    avg_total = (timing_sum['total'] / 50) * 1000
                    avg_batch_size = frame_count_sum / 50
                    throughput = frame_count_sum / timing_sum['total']

                    print(f"[BATCH PERF] Batch#{batch_count}: "
                          f"Avg={avg_batch_size:.1f} cams, "
                          f"Collect={avg_collect:.1f}ms, "
                          f"YOLO+ReID={avg_yolo:.1f}ms, "
                          f"Total={avg_total:.1f}ms, "
                          f"Throughput={throughput:.1f} FPS")

                    # Reset counters for next window
                    timing_sum = {'collect': 0, 'yolo': 0, 'total': 0}
                    frame_count_sum = 0

                # Step 3: Submit crops for exercise detection (if subclass overrides)
                for camera_id, reid_results in reid_results_dict.items():
                    record = records_dict.get(camera_id)
                    if record is not None:
                        frame = record["frame"]
                        # Call the subclass hook for crop submission (BridgeReIDService overrides this)
                        self._submit_crops_for_exercise(frame, reid_results, camera_id)

                # Process voting results after all crops submitted (if subclass has this method)
                if hasattr(self, 'process_voting_results'):
                    self.process_voting_results()

                # Step 4: Update persistent data and display for each camera
                for camera_id, reid_results in reid_results_dict.items():
                    # Update persistent detections
                    persistent_detections[camera_id] = reid_results
                    detection_timestamps[camera_id] = time.time()

                    # Update database stats periodically
                    self.update_database_stats()

                    # Get frame from records
                    record = records_dict.get(camera_id)
                    if record is None:
                        continue

                    frame = record["frame"]
                    camera = self.camera_manager.get_camera(camera_id)

                    if camera:
                        # Draw ReID results only if visual_mode is enabled
                        if self.visual_mode:
                            current_time = time.time()
                            if (current_time - detection_timestamps[camera_id]) < 2.0:
                                processed_frame = self.draw_reid_results(
                                    frame, persistent_detections[camera_id], camera_id
                                )
                            else:
                                processed_frame = self.draw_reid_results(frame, [], camera_id)

                            # Store processed frame
                            camera.set_processed_frame(processed_frame)
                        else:
                            # In non-visual mode, store original frame without drawing
                            camera.set_processed_frame(frame)

            except Exception as e:
                print(f"Error in batched processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def collect_frames_for_batch(self, timeout_ms=10):
        """
        Collect frames from all cameras for batch processing.

        Collects frames from all available cameras with a timeout to prevent
        waiting indefinitely. Returns whatever frames are ready within the timeout.

        Args:
            timeout_ms: Maximum time to wait for frame collection in milliseconds

        Returns:
            frames_dict: Dictionary {camera_id: frame} (may be partial if timeout)
            records_dict: Dictionary {camera_id: full_record} for metadata
        """
        frames_dict = {}
        records_dict = {}
        start_time = time.time()

        for camera_id in self.camera_manager.get_camera_ids():
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                break  # Timeout reached, process what we have

            camera = self.camera_manager.get_camera(camera_id)
            if camera:
                record = camera.get_frame()
                if record is not None:
                    frames_dict[camera_id] = record["frame"]
                    records_dict[camera_id] = record

        return frames_dict, records_dict

    def process_frames_batch_for_reid(self, frames_dict):
        """
        Process multiple frames for person re-identification using batch YOLO.

        This is the optimized version that processes multiple camera frames
        in a single YOLO inference for 6-9x speedup.

        Args:
            frames_dict: Dictionary {camera_id: frame}

        Returns:
            results_dict: Dictionary {camera_id: reid_results}
        """
        if not frames_dict:
            return {}

        try:
            # Step 1: Batch YOLO detection across all frames
            detection_results = self.person_detector.detect_persons_batch(frames_dict)

            # Step 2: Process each camera's detections (tracking + ReID)
            results_dict = {}
            for camera_id, (person_boxes, person_boxes_track) in detection_results.items():
                frame = frames_dict[camera_id]

                # Process tracking and ReID for this camera
                reid_results = self._process_tracking_and_reid(
                    frame, camera_id, person_boxes, person_boxes_track
                )
                results_dict[camera_id] = reid_results

            return results_dict

        except Exception as e:
            print(f"[ERROR] Batch YOLO processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {cam_id: [] for cam_id in frames_dict.keys()}

    def _process_tracking_and_reid(self, frame, camera_id, person_boxes, person_boxes_track):
        """
        Process tracking and ReID for a single camera after YOLO detection.

        This is extracted from process_frame_for_reid to enable batch YOLO
        while keeping per-camera tracking logic.

        Args:
            frame: Frame array
            camera_id: Camera identifier
            person_boxes: List of person detection dicts
            person_boxes_track: List of tracking format detections

        Returns:
            reid_results: List of ReID results for this frame
        """
        try:
            now = time.time()
            # Extract camera number from camera_id (e.g., 'cam_1' -> 1)
            sample_id = int(camera_id.split('_')[1]) - 1

            # Handle tracking reset logic
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

                        # Clear cache for this camera when tracking is reset
                        keys_to_remove = [key for key in self.track_id_to_person.keys() if key[0] == camera_id]
                        for key in keys_to_remove:
                            del self.track_id_to_person[key]

                    except Exception as e:
                        print(f"Error updating tracker with empty values: {e}")

                    self.last_detection_time[sample_id] = now

            if not person_boxes or not person_boxes_track:
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
                import traceback
                traceback.print_exc()
                return []

            # Process each target for ReID with database similarity search and caching
            for target in target_box:
                track_id, x, y, w, h, _, cam_id = target

                # Check cache first
                cached_person = self.get_cached_person(cam_id, track_id)

                if cached_person:
                    # Use cached result
                    person_name = cached_person['name']
                    similarity = cached_person['confidence']

                    # Update cache timestamp to keep it alive
                    self.update_cache_timestamp(cam_id, track_id)

                    # Convert coordinates
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    bbox = [x1, y1, x2, y2]
                    self.cache_master_list(cached_person['name'], cam_id, track_id)
                    # Create match object for consistency
                    match = {
                        'person_name': person_name,
                        'similarity': similarity,
                        'source_camera': 'cached'
                    } if person_name != "Unknown" else None

                    reid_results.append({
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'match': match,
                        'similarity': similarity,
                        'person_name': person_name,
                        'camera_id': cam_id,
                        'cached': True  # Flag to indicate this was from cache
                    })

                else:
                    # No cache hit, perform database ReID
                    # Convert from TLWH to TLBR
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)

                    # Get person crop using person detector utility
                    person_crop = self.person_detector.get_person_crop(frame, [x1, y1, x2, y2], padding=10)

                    if person_crop is None:
                        print(f"Empty crop for track_id {track_id}, skipping")
                        continue

                    try:
                        # Extract embedding using person embedder
                        embedding = self.person_embedder.extract_embedding(person_crop)

                        if embedding is not None:
                            # Database similarity search using similarity search utility
                            match, similarity = self.similarity_search.find_person_match_euclidean(
                                embedding, threshold=self.similarity_threshold
                            )

                            if match is not None:
                                person_name = match.get('person_name')
                                # Try to cache this result if confidence is high
                                self.cache_person_identification(cam_id, track_id, person_name, similarity)
                            else:
                                person_name = "Unknown"
                                # Cache "Unknown" only if we're very confident it's not a match
                                if similarity < 0.3:  # Very low similarity, likely not anyone we know
                                    self.cache_person_identification(cam_id, track_id, person_name, 1.0 - similarity)

                            # Store detailed results
                            reid_results.append({
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2],
                                'match': match,
                                'similarity': similarity,
                                'person_name': person_name,
                                'camera_id': cam_id,
                                'cached': False  # Flag to indicate this was freshly computed
                            })
                        else:
                            print(f"Track ID {track_id}: Failed to extract embedding")

                    except Exception as e:
                        print(f"Error processing track_id {track_id}: {e}")
                        continue

            return reid_results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return []

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

        # Draw person re-identification results
        for result in reid_results:
            bbox = result['bbox']
            match = result['match']
            similarity = result['similarity']
            track_id = result.get('track_id', 'N/A')
            person_name = result.get('person_name', 'Unknown')
            is_cached = result.get('cached', False)
            exercise_info = result.get('exercise', None)  # Get exercise prediction

            # Scale and offset coordinates
            x1 = int(bbox[0] * scale_x) + pad_left
            y1 = int(bbox[1] * scale_y) + pad_top
            x2 = int(bbox[2] * scale_x) + pad_left
            y2 = int(bbox[3] * scale_y) + pad_top

            # Choose color and label based on match and cache status
            if match and similarity >= self.similarity_threshold:
                if is_cached:
                    color = (0, 255, 0)  # green for cached matches
                    cache_indicator = " [CACHED]"
                else:
                    color = (0, 255, 225)    # yellow for fresh database matches
                cache_indicator = " [DB]"

                label = f"ID:{track_id} {person_name} ({similarity:.2f}){cache_indicator}"
                if match.get('source_camera') not in ['unknown', 'cached']:
                    label += f" [{match['source_camera']}]"
            else:
                if is_cached:
                    color = (128, 0, 128)  # Purple for cached unknowns
                    cache_indicator = " [CACHED]"
                else:
                    color = (0, 0, 255)    # Red for fresh database unknowns
                    cache_indicator = " [DB]"

                label = f"ID:{track_id} Unknown ({similarity:.2f}){cache_indicator}"

            # Draw bounding box with thicker line for cached results
            line_thickness = 3 if is_cached else 2
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, line_thickness)

            # Draw tracking ID prominently at top-left of box
            track_id_text = f"ID:{track_id}"
            track_id_size = cv2.getTextSize(track_id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Background color for tracking ID (different for cached vs database)
            bg_color = (255, 255, 0) if not is_cached else (0, 255, 255)  # Yellow for DB, Cyan for cached
            cv2.rectangle(display_frame, (x1, y1 - track_id_size[1] - 15),
                        (x1 + track_id_size[0] + 10, y1 - 5), bg_color, -1)
            cv2.putText(display_frame, track_id_text, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw main label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y2),
                        (x1 + label_size[0], y2 + label_size[1] + 10), color, -1)
            cv2.putText(display_frame, label, (x1, y2 + label_size[1] + 5),
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
        cache_size = len(self.track_id_to_person)
        cache_info = f"Cache Size: {cache_size}"
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
                    current_frame_count = camera.get_frame_count()

                    # Only update FPS if we have a new frame (not the same frame as before)
                    if processed_frame is not None:
                        self.stable_display_buffer[camera_id] = processed_frame
                        updated = True

                        # Only calculate FPS if this is a NEW frame
                        if current_frame_count != self.last_frame_count.get(camera_id, -1):
                            self.calculate_fps(camera_id)
                            self.last_frame_count[camera_id] = current_frame_count

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

    def create_multi_window_display(self, camera_ids_chunk, window_index):
        """
        Create a grid display for a subset of cameras (one window).

        Args:
            camera_ids_chunk (list): List of camera IDs to display in this window
            window_index (int): Index of the window (for labeling)

        Returns:
            numpy.ndarray: Combined grid display for this window's cameras
        """
        # Calculate grid dimensions for this window
        num_cams = len(camera_ids_chunk)
        grid_cols = 3  # 3x2 grid for 6 cameras per window
        grid_rows = int(np.ceil(num_cams / grid_cols))

        window_width = grid_cols * self.camera_width
        window_height = grid_rows * self.camera_height

        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        with self.display_lock:
            for i, camera_id in enumerate(camera_ids_chunk):
                # Calculate grid position within this window
                row = i // grid_cols
                col = i % grid_cols

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
        self.camera_manager.print_first_pts()
        self.camera_manager.sample_pts_delta()

        # Start processing threads - batched or per-camera based on configuration
        self.processing_threads = []
        if self.use_batch_yolo:
            # Single batched processing thread for all cameras (6-9x faster YOLO)
            print(f"[BATCH MODE] Starting single batched processing loop for all {self.num_cameras} cameras")
            thread = threading.Thread(target=self.batched_processing_loop)
            thread.daemon = True
            thread.start()
            self.processing_threads.append(('batch_all', thread))
        else:
            # Legacy: One thread per camera (for backward compatibility)
            print(f"[LEGACY MODE] Starting {self.num_cameras} individual processing threads")
            for camera_id in self.camera_manager.get_camera_ids():
                thread = threading.Thread(target=self.processing_thread, args=(camera_id,))
                thread.daemon = True
                thread.start()
                self.processing_threads.append((camera_id, thread))
        
        # Split cameras into windows (6 cameras per window)
        cameras_per_window = 6
        all_camera_ids = self.camera_manager.get_camera_ids()
        camera_chunks = [all_camera_ids[i:i + cameras_per_window]
                        for i in range(0, len(all_camera_ids), cameras_per_window)]

        print(f"Creating {len(camera_chunks)} windows for {self.num_cameras} cameras")
        for i, chunk in enumerate(camera_chunks):
            print(f"  Window {i+1}: {len(chunk)} cameras ({chunk[0]} to {chunk[-1]})")

        try:
            while self.running:
                # Update display buffer
                self.update_stable_display()

                # Only display if visual_mode is enabled
                if self.visual_mode and not os.getenv('HEADLESS'):
                    # Create and show display for each window
                    for window_idx, camera_chunk in enumerate(camera_chunks):
                        window_display = self.create_multi_window_display(camera_chunk, window_idx)
                        window_name = f'Person Re-ID - Window {window_idx + 1} (Cameras {camera_chunk[0]} to {camera_chunk[-1]})'
                        cv2.imshow(window_name, window_display)

                # Cleanup expired cache entries periodically
                if time.time() % 10 < 0.1:  # Every ~10 seconds
                    self.cleanup_expired_cache_entries()

                # Handle controls
                if self.visual_mode:
                    key = cv2.waitKey(33) & 0xFF
                else:
                    # In non-visual mode, just sleep to prevent busy loop
                    time.sleep(0.033)  # ~30 fps equivalent
        
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