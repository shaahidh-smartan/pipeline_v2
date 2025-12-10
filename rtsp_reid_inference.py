#!/usr/bin/env python3
"""
RTSP Stream Person Re-Identification using BPBreID and ByteTrack.
Real-time inference on RTSP camera streams with voting-based caching.
"""
import os
import sys
import argparse
import time
import signal
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add ByteTrack to path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, 'ByteTrack'))

from torchreid.scripts.main import build_config
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.utils.constants import bn_correspondants

# Import ByteTrack
from yolox.tracker.byte_tracker import BYTETracker


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


class RTSPReIDInference:
    """RTSP Stream ReID Inference Pipeline with voting-based caching."""

    def __init__(self, config_path, weights_path, gallery_dir, rtsp_url,
                 voting_threshold=4.5, voting_window=30, min_votes=15,
                 matching_threshold=6.0, frame_skip=0, reconnect_delay=5.0,
                 yolo_conf=0.5, yolo_min_height=50, yolo_min_width=20,
                 top_k=1, show_all_ranks=False,
                 device='cuda'):
        """
        Initialize RTSP ReID Inference with voting-based caching.

        Args:
            config_path: Path to BPBreID config YAML
            weights_path: Path to model weights
            gallery_dir: Directory with gallery embeddings
            rtsp_url: RTSP stream URL
            voting_threshold: Distance threshold for counting a vote (default 4.5)
            voting_window: Number of frames to collect votes (default 30)
            min_votes: Minimum votes needed to cache an ID (default 15)
            matching_threshold: Distance threshold for matching/display (default 6.0)
            frame_skip: Skip N frames between processing (default 0 = process all)
            reconnect_delay: Seconds to wait before reconnecting (default 5.0)
            yolo_conf: YOLO confidence threshold (default 0.5)
            yolo_min_height: Minimum person height in pixels (default 50)
            yolo_min_width: Minimum person width in pixels (default 20)
            top_k: Number of top matches to consider (default 1)
            show_all_ranks: Show all top-k matches in display (default False)
            device: 'cuda' or 'cpu'
        """
        self.rtsp_url = rtsp_url
        self.voting_threshold = voting_threshold
        self.voting_window = voting_window
        self.min_votes = min_votes
        self.matching_threshold = matching_threshold
        self.frame_skip = frame_skip
        self.reconnect_delay = reconnect_delay
        self.yolo_conf = yolo_conf
        self.yolo_min_height = yolo_min_height
        self.yolo_min_width = yolo_min_width
        self.top_k = top_k
        self.show_all_ranks = show_all_ranks
        self.device = device

        # Stream info (will be set when connected)
        self.cap = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.is_connected = False

        # Initialize ReID components
        self._initialize_reid(config_path, weights_path, gallery_dir, device)

        # Camera ID
        self.camera_id = rtsp_url
        self.frame_count = 0
        self.processed_frame_count = 0

        # Graceful shutdown flag
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)

        print("[INIT] ✓ RTSP ReID Inference initialized\n")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C for graceful shutdown."""
        print("\n\n[SHUTDOWN] Received interrupt signal, stopping gracefully...")
        self.should_stop = True

    def _connect_stream(self):
        """Connect to RTSP stream."""
        print(f"[RTSP] Connecting to {self.rtsp_url}...")

        self.cap = cv2.VideoCapture(self.rtsp_url)

        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"[ERROR] Failed to connect to RTSP stream")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25 if unknown
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_connected = True

        print(f"[RTSP] ✓ Connected: {self.width}x{self.height} @ {self.fps:.1f}fps")
        return True

    def _disconnect_stream(self):
        """Disconnect from RTSP stream."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False

    def _initialize_reid(self, config_path, weights_path, gallery_dir, device):
        """Initialize BPBreID and ByteTrack."""
        print("\n[INIT] Loading BPBreID model...")

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

        # Load gallery
        print("\n[INIT] Loading gallery...")
        gallery_path = Path(gallery_dir)
        self.gallery_embeddings = torch.load(gallery_path / 'gallery_embeddings.pt', map_location=self.device)
        self.gallery_visibility = torch.load(gallery_path / 'gallery_visibility.pt', map_location=self.device)
        self.gallery_pids = torch.load(gallery_path / 'gallery_pids.pt', map_location=self.device)

        print(f"        Gallery: {self.gallery_embeddings.shape[0]} samples, "
              f"{self.gallery_embeddings.shape[1]} parts, PIDs: {torch.unique(self.gallery_pids).tolist()}")

        # Load YOLO detector
        print("\n[INIT] Loading YOLO detector...")
        self.yolo = YOLO('yolo11n.pt')

        # Initialize ByteTracker (will be reset per connection)
        self.tracker = None

        # Voting-based cache structures
        self.track_voting = {}
        self.cached_tracks = {}

    def _initialize_tracker(self):
        """Initialize ByteTracker with current stream FPS."""
        print(f"\n[INIT] Initializing ByteTracker with FPS={self.fps:.1f}...")
        tracker_args = type('Args', (object,), {
            'track_thresh': 0.5,
            'match_thresh': 0.6,
            'track_buffer': 30,
            'mot20': False,
            'frame_rate': int(self.fps)
        })
        self.tracker = BYTETracker(tracker_args, frame_rate=int(self.fps))

        # Reset tracking data
        self.track_voting = {}
        self.cached_tracks = {}

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

    def process_frame(self, frame):
        """Process single frame for ReID."""
        self.processed_frame_count += 1
        current_time = time.time()

        # Run YOLO detection
        results = self.yolo(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls != 0:  # Skip non-person
                    continue

                conf = float(boxes.conf[i])
                if conf < self.yolo_conf:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                if (y2 - y1) < self.yolo_min_height or (x2 - x1) < self.yolo_min_width:
                    continue

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                detections.append([x1, y1, x2, y2, conf, 1.0])

        reid_results = []
        if not detections:
            return reid_results

        # Update ByteTracker
        online_targets = self.tracker.update(
            np.array(detections, dtype=np.float64),
            [self.height, self.width],
            [self.height, self.width]
        )

        # Process each tracked person
        for track in online_targets:
            track_id = track.track_id
            tlwh = track.tlwh
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h

            # Crop for ReID
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Check if this track is already cached
            if track_id in self.cached_tracks:
                person_name = self.cached_tracks[track_id]['person_id']
                result_distance = self.cached_tracks[track_id]['distance']
                is_cached = True

            else:
                # Extract embeddings
                model_output = self.extractor([crop_rgb])
                embeddings, visibility = self.extract_test_embeddings(model_output)
                embeddings = embeddings.to(self.device)
                visibility = visibility.to(self.device)

                # Compute distances
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

                # Get top-k matches
                distances = distmat[0].cpu().numpy()
                top_k_indices = np.argsort(distances)[:self.top_k]
                top_k_matches = [(int(self.gallery_pids[idx].item()), distances[idx])
                                for idx in top_k_indices]

                # Use best match for tracking/caching
                best_pid, best_dist = top_k_matches[0]
                person_name = str(best_pid)
                result_distance = best_dist
                is_cached = False

                # Voting logic
                if track_id not in self.track_voting:
                    self.track_voting[track_id] = {
                        'votes': {},
                        'frame_count': 0,
                        'distances': {}
                    }

                voting_entry = self.track_voting[track_id]
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
                            self.cached_tracks[track_id] = {
                                'person_id': max_votes_id,
                                'distance': voting_entry['distances'][max_votes_id],
                                'votes': max_votes
                            }
                            print(f"[CACHE] T{track_id} -> ID{max_votes_id} "
                                  f"(votes={max_votes}/{self.voting_window})")

                            person_name = max_votes_id
                            result_distance = voting_entry['distances'][max_votes_id]
                            is_cached = True

                    del self.track_voting[track_id]

            # Build result
            result = {
                'bbox': (x1, y1, x2, y2),
                'track_id': track_id,
                'person_name': person_name,
                'distance': result_distance,
                'cached': is_cached
            }

            # Add top-k matches if available (not cached)
            if not is_cached and 'top_k_matches' in locals():
                result['top_k_matches'] = top_k_matches

            reid_results.append(result)

        # Cleanup disappeared tracks
        active_track_ids = {track.track_id for track in online_targets}
        disappeared_tracks = set(self.cached_tracks.keys()) - active_track_ids
        for track_id in disappeared_tracks:
            del self.cached_tracks[track_id]

        disappeared_voting = set(self.track_voting.keys()) - active_track_ids
        for track_id in disappeared_voting:
            del self.track_voting[track_id]

        return reid_results

    def draw_results(self, frame, reid_results):
        """Draw ReID results on frame."""
        annotated = frame.copy()

        for result in reid_results:
            x1, y1, x2, y2 = result['bbox']
            person_name = result['person_name']
            track_id = result['track_id']
            distance = result['distance']
            is_cached = result['cached']

            # Determine if recognized
            is_recognized = distance < self.matching_threshold

            # Color: Blue (cached), Green (match), Red (unknown)
            if is_cached:
                color = (255, 0, 0)  # Blue
                label = f"T{track_id}|ID{person_name}* d={distance:.2f}"
            elif is_recognized:
                color = (0, 255, 0)  # Green
                label = f"T{track_id}|ID{person_name} d={distance:.2f}"
            else:
                color = (0, 0, 255)  # Red
                label = f"T{track_id}|UNK d={distance:.2f}"

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw main label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw top-k rankings if enabled
            if self.show_all_ranks and 'top_k_matches' in result and len(result['top_k_matches']) > 1:
                y_offset = y2 + 20
                for rank, (pid, dist) in enumerate(result['top_k_matches'], 1):
                    rank_label = f"  #{rank}: ID{pid} d={dist:.2f}"
                    cv2.putText(annotated, rank_label, (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 18

        # Frame info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"{timestamp} | Processed: {self.processed_frame_count} | Cached: {len(self.cached_tracks)}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Connection status
        status = "LIVE" if self.is_connected else "DISCONNECTED"
        status_color = (0, 255, 0) if self.is_connected else (0, 0, 255)
        cv2.circle(annotated, (self.width - 30, 30), 10, status_color, -1)
        cv2.putText(annotated, status, (self.width - 120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return annotated

    def run(self, output_path=None, display=True):
        """Run RTSP inference with auto-reconnection."""
        print(f"[START] Starting RTSP inference...")
        print(f"        Output: {output_path or 'None (display only)'}")
        print(f"        Frame skip: {self.frame_skip}")
        print(f"        Press Ctrl+C to stop\n")

        out_writer = None
        process_times = []
        start_time = time.time()
        last_fps_update = time.time()
        fps_counter = 0
        current_fps = 0.0

        try:
            while not self.should_stop:
                # Connect if not connected
                if not self.is_connected:
                    if not self._connect_stream():
                        print(f"[RECONNECT] Retrying in {self.reconnect_delay}s...")
                        time.sleep(self.reconnect_delay)
                        continue

                    # Initialize tracker with stream FPS
                    self._initialize_tracker()

                    # Setup output writer if needed
                    if output_path and not out_writer:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_writer = cv2.VideoWriter(output_path, fourcc, self.fps,
                                                     (self.width, self.height))

                # Read frame
                ret, frame = self.cap.read()

                if not ret:
                    print("[ERROR] Failed to read frame, reconnecting...")
                    self._disconnect_stream()
                    continue

                self.frame_count += 1

                # Skip frames if needed
                if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1) != 0):
                    continue

                # Process frame
                frame_start = time.time()
                reid_results = self.process_frame(frame)
                annotated = self.draw_results(frame, reid_results)
                frame_time = time.time() - frame_start
                process_times.append(frame_time)

                # Calculate FPS
                fps_counter += 1
                if time.time() - last_fps_update >= 1.0:
                    current_fps = fps_counter / (time.time() - last_fps_update)
                    fps_counter = 0
                    last_fps_update = time.time()

                # Add FPS to display
                cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Write output
                if out_writer:
                    out_writer.write(annotated)

                # Display
                if display:
                    cv2.imshow('RTSP ReID', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        print("\n[STOP] User requested stop")
                        break

                # Print stats periodically
                if self.processed_frame_count % 100 == 0:
                    avg_time = sum(process_times[-100:]) / min(100, len(process_times))
                    print(f"[STATS] Processed: {self.processed_frame_count}, "
                          f"FPS: {current_fps:.1f}, "
                          f"Avg time: {avg_time*1000:.1f}ms, "
                          f"Cached: {len(self.cached_tracks)}")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._disconnect_stream()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_time = sum(process_times) / len(process_times) if process_times else 0
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0

            print(f"\n\n[STATS] Session complete")
            print(f"        Total frames: {self.frame_count}")
            print(f"        Processed frames: {self.processed_frame_count}")
            print(f"        Total time: {total_time:.1f}s")
            print(f"        Avg processing FPS: {avg_fps:.1f}")
            print(f"        Final cached tracks: {len(self.cached_tracks)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RTSP BPBreID ReID Inference')
    parser.add_argument('--config', type=str, required=True, help='BPBreID YAML config')
    parser.add_argument('--weights', type=str, default=None, help='Model weights path')
    parser.add_argument('--gallery-dir', type=str, required=True, help='Gallery directory')
    parser.add_argument('--rtsp-url', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--output', type=str, default=None, help='Output video path (optional)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # Voting and caching parameters
    parser.add_argument('--voting-threshold', type=float, default=4.5,
                       help='Distance threshold for votes (default: 4.5)')
    parser.add_argument('--voting-window', type=int, default=30,
                       help='Frames to collect votes (default: 30)')
    parser.add_argument('--min-votes', type=int, default=15,
                       help='Minimum votes to cache (default: 15)')
    parser.add_argument('--matching-threshold', type=float, default=6.0,
                       help='Display matching threshold - distance > threshold = Unknown (default: 6.0)')

    # YOLO detection parameters
    parser.add_argument('--yolo-conf', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--yolo-min-height', type=int, default=50,
                       help='Minimum person height in pixels (default: 50)')
    parser.add_argument('--yolo-min-width', type=int, default=20,
                       help='Minimum person width in pixels (default: 20)')

    # Ranking and display parameters
    parser.add_argument('--top-k', type=int, default=1,
                       help='Number of top matches to consider (default: 1)')
    parser.add_argument('--show-all-ranks', action='store_true',
                       help='Display all top-k rankings on video (default: False)')

    # Performance and connection parameters
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='Skip N frames (0=process all, default: 0)')
    parser.add_argument('--reconnect-delay', type=float, default=5.0,
                       help='Reconnection delay in seconds (default: 5.0)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')

    args = parser.parse_args()

    print("=" * 80)
    print("RTSP STREAM PERSON RE-IDENTIFICATION (VOTING-BASED)")
    print("=" * 80)

    try:
        inference = RTSPReIDInference(
            config_path=args.config,
            weights_path=args.weights,
            gallery_dir=args.gallery_dir,
            rtsp_url=args.rtsp_url,
            voting_threshold=args.voting_threshold,
            voting_window=args.voting_window,
            min_votes=args.min_votes,
            matching_threshold=args.matching_threshold,
            frame_skip=args.frame_skip,
            reconnect_delay=args.reconnect_delay,
            yolo_conf=args.yolo_conf,
            yolo_min_height=args.yolo_min_height,
            yolo_min_width=args.yolo_min_width,
            top_k=args.top_k,
            show_all_ranks=args.show_all_ranks,
            device=args.device
        )

        inference.run(
            output_path=args.output,
            display=not args.no_display
        )

        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
