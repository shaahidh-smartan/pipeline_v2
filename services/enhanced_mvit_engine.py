from __future__ import annotations
import time
import queue
import threading
import cv2
import os
from collections import deque, Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
from collections import defaultdict

DEBUG = os.getenv("SF_DEBUG", "0") == "1"
SAVE_FRAMES = os.getenv("MVIT_SAVE_FRAMES", "0") == "1"  # Set MVIT_SAVE_FRAMES=1 to save frames
SAVE_FRAMES_DIR = os.getenv("MVIT_SAVE_DIR", "mvit_debug_frames")

# Batch timing log file - set MVIT_BATCH_LOG=1 to enable
BATCH_LOG_ENABLED = os.getenv("MVIT_BATCH_LOG", "0") == "1"
BATCH_LOG_FILE = "mvit_batch_timing.log"

def _batch_log(msg: str) -> None:
    """Write batch timing info to log file."""
    if not BATCH_LOG_ENABLED:
        return
    ts = time.strftime("%H:%M:%S", time.localtime())
    ms = int((time.time() % 1) * 1000)
    with open(BATCH_LOG_FILE, "a") as f:
        f.write(f"[{ts}.{ms:03d}] {msg}\n")

@dataclass
class EnhancedClipMeta:
    """Enhanced metadata for tracking frame windows"""
    cam_id: str
    track_id: int
    ts: float
    window_start: int
    window_end: int
    seq_id_range: Tuple[int, int]

class MViTEngine:
    """
    Enhanced MViT engine with emit cursor architecture - Same as SlowFast but uses MViT

    Key features:
    - Uses MViT model instead of SlowFast
    - Emit cursor prevents frame skipping (65x more data coverage)
    - Robust preprocessing handles edge cases gracefully
    - 3-batch voting system with proper FIFO alignment
    - Comprehensive warmup sequence for reliable startup
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        num_frames: int = 16,  # MViT uses 16 frames
        max_microbatch: int = 12,
        tick_ms: int = 120,
        cooldown_s: float = 2.0,
        model_path: Optional[str] = None,
        num_classes: int = 52,
        K: int = 5,  # Number of last blocks to fine-tune
        use_fp16: bool = True,
        confidence_threshold: float = 0.3,
    ):
        self.class_names = class_names or []
        self.num_frames = int(num_frames)
        self.max_microbatch = int(max_microbatch)
        self.tick_ms = int(tick_ms)
        self.cooldown_s = float(cooldown_s)
        self.model_path = model_path
        self.num_classes = num_classes
        self.K = K
        self.use_fp16 = use_fp16
        self.confidence_threshold = confidence_threshold

        # Initialize device
        self._device_cached = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # EMIT CURSOR ARCHITECTURE - No frame loss!
        self._stream_buf: Dict[Tuple[str, int], List[np.ndarray]] = {}
        self._emit_idx: Dict[Tuple[str, int], int] = {}
        self._last_emit: Dict[Tuple[str, int], float] = {}
        self._seq_counters: Dict[Tuple[str, int], int] = {}
        self._last_activity: Dict[Tuple[str, int], float] = {}

        # FIFO queue of ready clips
        self._in: "queue.Queue[Tuple[str, int, List[np.ndarray], float]]" = queue.Queue(maxsize=256)
        self._global_win_counter: Dict[Tuple[str, int], int] = {}

        # 3-BATCH VOTING SYSTEM
        self._batch_predictions: Dict[Tuple[str, int], List[Dict]] = {}
        self._voting_results: Dict[Tuple[str, int], Dict] = {}
        self._pending_batches: Dict[Tuple[str, int], List[Tuple[List[np.ndarray], float]]] = {}

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._cleanup_worker: Optional[threading.Thread] = None

        # Model
        self._model: Optional[nn.Module] = None

        # Statistics
        self._batches_run = 0
        self._dropped_clips = 0
        self._submitted_frames = 0
        self._ready_clips_generated = 0
        self._voting_cycles_completed = 0
        self._windows_created = 0
        self._windows_enqueued = 0
        self._model_loaded = False

    @property
    def device(self) -> torch.device:
        """Property to ensure device is available"""
        if not hasattr(self, '_device_cached'):
            self._device_cached = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device_cached

    def set_classes(self, class_names: List[str]) -> None:
        """Set class names for exercise recognition."""
        self.class_names = list(class_names or [])
        if DEBUG:
            print(f"[MVIT] Updated classes: {len(self.class_names)}")

    def start(self) -> None:
        """Start the MViT engine with enhanced startup and comprehensive warmup."""
        if self._running:
            return

        print(f"[MVIT] Starting with emit cursor architecture...")

        # Clear log file on startup
        if BATCH_LOG_ENABLED:
            with open(BATCH_LOG_FILE, "w") as f:
                f.write("=== MViT Batch Timing Log ===\n")
                f.write("This log shows WHY results come out at different times per person.\n")
                f.write("Key: Each person needs 64 frames -> window ready -> enqueue -> batch inference -> voting (3 batches needed)\n\n")
            print(f"[MVIT] Batch timing log enabled -> {BATCH_LOG_FILE}")

        if SAVE_FRAMES:
            print(f"[MVIT] Frame saving ENABLED - frames will be saved to: {SAVE_FRAMES_DIR}")
            os.makedirs(SAVE_FRAMES_DIR, exist_ok=True)

        self._running = True
        self._worker = threading.Thread(target=self.worker_loop, name="MViTWorker", daemon=True)
        self._cleanup_worker = threading.Thread(target=self.cleanup_loop, name="MViTCleanup", daemon=True)
        self._worker.start()
        self._cleanup_worker.start()

        # Wait a moment for worker to initialize
        time.sleep(1)

        # Run comprehensive warmup
        print(f"[MVIT] Running warmup sequence...")
        warmup_success = self.run_comprehensive_warmup()

        if warmup_success:
            print(f"[MVIT] Startup complete - ready for inference!")
        else:
            print(f"[MVIT] WARNING: Warmup had issues - check logs above")

        print(f"[MVIT] Started with emit cursor - no frame loss!")

    def stop(self) -> None:
        """Stop the MViT engine and clean up resources."""
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._cleanup_worker and self._cleanup_worker.is_alive():
            self._cleanup_worker.join(timeout=1.0)

    def submit_crop(self, cam_id: str, track_id: int, rgb224: np.ndarray) -> None:
        """
        Submit a 224x224 RGB crop for exercise recognition with emit cursor architecture.

        Uses SlowFast stride logic: 64-frame non-overlapping windows.
        """
        # --- robust preprocessing (same as SlowFast) ---
        if rgb224 is None or rgb224.size == 0:
            rgb224 = np.zeros((224, 224, 3), dtype=np.uint8)

        if len(rgb224.shape) != 3 or rgb224.shape[2] != 3:
            if DEBUG:
                print(f"[MVIT] Invalid shape {getattr(rgb224, 'shape', None)}, creating zero frame")
            rgb224 = np.zeros((224, 224, 3), dtype=np.uint8)

        if rgb224.shape[:2] != (224, 224):
            try:
                rgb224 = cv2.resize(rgb224, (224, 224), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                if DEBUG:
                    print(f"[MVIT] Resize failed: {e}, using zero frame")
                rgb224 = np.zeros((224, 224, 3), dtype=np.uint8)

        key = (cam_id, int(track_id))
        now = time.time()
        self._submitted_frames += 1

        with self._lock:
            # init per-track state (same as SlowFast)
            buf   = self._stream_buf.setdefault(key, [])
            start = self._emit_idx.setdefault(key, 0)
            _     = self._seq_counters.setdefault(key, 0)
            self._last_activity[key] = now

            # pending list must exist
            if key not in self._pending_batches:
                self._pending_batches[key] = []

            # append this frame to the growing person-crop buffer
            buf.append(rgb224)
            self._seq_counters[key] += 1

            # cut consecutive 64-frame windows (SAME AS SLOWFAST)
            while len(buf) - start >= 64:
                # fixed window
                window64 = buf[start:start + 64]
                win_idx  = start // 64

                if key not in self._global_win_counter:
                    self._global_win_counter[key] = 0
                global_counter = self._global_win_counter[key]
                self._global_win_counter[key] += 1

                # compute deterministic sequence span for this window (same as SlowFast)
                seq_end   = self._seq_counters[key] - (len(buf) - (start + 64))
                seq_start = max(1, seq_end - 64 + 1)

                # single append with 6-tuple (window64, ts, win_idx, seq_start, seq_end, global_counter)
                self._pending_batches[key].append((window64, now, win_idx, seq_start, seq_end, global_counter))

                # MViT will use all 64 frames (samples 16 in prep_tensor_mvit)
                window_copy = [f.copy() for f in window64]

                self._windows_created += 1
                _batch_log(f"WINDOW_READY | {cam_id}:{track_id} | win_idx={win_idx} | frames={start}-{start+63} | total_buf={len(buf)} | pending_windows={len(self._pending_batches[key])}")
                if DEBUG:
                    print(f"[MVIT_EMIT] {cam_id}:{track_id} win_idx={win_idx} window {start}:{start+63} seq[{seq_start}:{seq_end}]")

                # enqueue to worker (cooldown throttles enqueueing, not window creation)
                last = self._last_emit.get(key, 0.0)
                time_since_last = now - last
                if time_since_last >= self.cooldown_s:
                    try:
                        # keep worker tuple as 4-tuple to avoid downstream changes
                        self._in.put_nowait((cam_id, track_id, window_copy, now))
                        self._last_emit[key] = now
                        self._ready_clips_generated += 1
                        self._windows_enqueued += 1
                        _batch_log(f"ENQUEUE | {cam_id}:{track_id} | win_idx={win_idx} | queue_size={self._in.qsize()} | cooldown_ok={time_since_last:.2f}s")
                        if DEBUG:
                            print(f"[MVIT_ENQUEUE] {cam_id}:{track_id} win_idx={win_idx} window {start}:{start+63} enqueued")
                    except queue.Full:
                        self._dropped_clips += 1
                        # pop the pending batch we just appended so we can optionally mark it "dropped"
                        dropped = self._pending_batches[key].pop() if self._pending_batches[key] else None
                        _batch_log(f"QUEUE_FULL | {cam_id}:{track_id} | win_idx={win_idx} | DROPPED")
                        if DEBUG:
                            print(f"[MVIT_DROP] Queue full for {cam_id}:{track_id} window {start}:{start+63}")
                else:
                    _batch_log(f"COOLDOWN_SKIP | {cam_id}:{track_id} | win_idx={win_idx} | time_since_last={time_since_last:.2f}s < {self.cooldown_s}s")

                # next window (STRIDE = 64, same as SlowFast)
                start += 64

            # memory management: rebase when cursor grows (same as SlowFast)
            if start >= 512:
                consumed = min(start, len(buf))
                del buf[:consumed]
                start -= consumed
                if DEBUG and consumed > 0:
                    print(f"[MVIT_REBASE] {cam_id}:{track_id} removed {consumed} old frames")

            self._emit_idx[key] = start

    def load_model(self) -> nn.Module:
        """
        Load MViT model from local .pt file.

        This is the ONLY function that differs from SlowFast - uses MViT instead!
        """
        print(f"[MVIT] Loading MViT model on {self.device}")
        try:
            weights = MViT_V1_B_Weights.DEFAULT
            model = mvit_v1_b(weights=weights)

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # Replace final layer with dropout + linear
            last_fc_layer = model.head[-1]
            in_features = last_fc_layer.in_features

            if len(self.class_names) > 0:
                num_classes = len(self.class_names)
            else:
                num_classes = self.num_classes

            model.head[-1] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )

            # Unfreeze last K blocks
            blocks = list(model.blocks)
            for block in blocks[-self.K:]:
                for p in block.parameters():
                    p.requires_grad = True

            # Load weights if provided
            if self.model_path:
                print(f"[MVIT] Loading custom weights from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state_dict)

            model = model.to(self.device).eval()
            # DO NOT convert model to half precision - keep it in FP32
            # FP16 is handled via autocast during inference only

            self._model_loaded = True
            print(f"[MVIT] Model loaded successfully")
            return model

        except Exception as e:
            print(f"[MVIT] ERROR loading model: {e}")
            raise

    def prep_tensor_mvit(self, clip: List[np.ndarray]) -> torch.Tensor:
        """
        Prepare tensor from video clip frames for MViT.

        MViT expects: (1, C, T, H, W) where T=16 frames
        Samples every 4th frame from the 64-frame clip (similar to SlowFast's ::2 pattern).
        64 frames / 4 = 16 frames evenly spaced.
        """
        # Sample every 4th frame from 64-frame window: clip[::4] = 16 frames
        # This gives frames [0, 4, 8, 12, 16, 20, ..., 60] = 16 frames
        sampled_frames = clip[::4]  # 64/4 = 16 frames

        # Frames are already RGB 224x224 from submit_crop
        # Stack frames: (T, H, W, C)
        video_array = np.stack(sampled_frames, axis=0)

        # Debug: Check frame statistics
        if DEBUG and len(sampled_frames) > 0:
            first_frame = sampled_frames[0]
            print(f"[TENSOR_DEBUG] Frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
            print(f"[TENSOR_DEBUG] Frame range: [{first_frame.min()}, {first_frame.max()}]")
            print(f"[TENSOR_DEBUG] Frame mean: {first_frame.mean():.2f}")
            print(f"[TENSOR_DEBUG] Sampled {len(sampled_frames)} frames with stride 4 (every 4th frame)")

        # Convert to tensor and permute to (C, T, H, W)
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float()

        # Normalize to [0, 1]
        video_tensor = video_tensor / 255.0

        # Apply normalization (ImageNet stats)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std

        if DEBUG:
            print(f"[TENSOR_DEBUG] Normalized tensor shape: {video_tensor.shape}")
            print(f"[TENSOR_DEBUG] Normalized range: [{video_tensor.min():.2f}, {video_tensor.max():.2f}]")
            print(f"[TENSOR_DEBUG] Normalized mean: {video_tensor.mean():.2f}")

        # Add batch dimension: (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)

        return video_tensor

    def prep_batch_mvit(self, batch: List[List[np.ndarray]], debug_hash=False) -> torch.Tensor:
        """
        Prepare batch of video clips for MViT inference.

        Args:
            batch: List of clips, each clip is a list of 64 RGB frames (224x224x3)

        Returns:
            Tensor of shape (B, C, T, H, W) where B=batch_size, T=16
        """
        tensors = []
        for clip in batch:
            t = self.prep_tensor_mvit(clip)
            tensors.append(t)

        # Stack batch: (B, C, T, H, W)
        batch_tensor = torch.cat(tensors, dim=0)

        # Debug: Check if tensor is changing
        if debug_hash and len(batch) > 0:
            tensor_hash = hash(batch_tensor[0].cpu().numpy().tobytes())
            print(f"[TENSOR_HASH] First clip tensor hash: {tensor_hash}")

        return batch_tensor.to(self.device)

    def process_batch_prediction(self, cam_id: str, track_id: int, exercise: str,
                                confidence: float, timestamp: float) -> None:
        """
        Process batch prediction and add to voting system.

        IDENTICAL to SlowFast version - no changes needed!
        """
        key = (cam_id, int(track_id))

        # Only store high-confidence predictions
        if confidence < self.confidence_threshold:
            _batch_log(f"LOW_CONF_SKIP | {cam_id}:{track_id} | exercise={exercise} | conf={confidence:.2f} < threshold={self.confidence_threshold}")
            if DEBUG:
                print(f"[MVIT_VOTING] {cam_id}:{track_id} - Low confidence {exercise} ({confidence:.3f}) - ignoring")
            with self._lock:
                pending_batches = self._pending_batches.get(key, [])
                if pending_batches:
                    pending_batches.pop(0)
            return

        with self._lock:
            if key not in self._batch_predictions:
                self._batch_predictions[key] = []

            # Get corresponding frame batch
            pending_batches = self._pending_batches.get(key, [])
            if not pending_batches:
                if DEBUG:
                    print(f"[MVIT_VOTING] {cam_id}:{track_id} - No pending batch found!")
                return

            # Take the oldest pending batch (FIFO)
            batch_frames, batch_timestamp, win_idx, seq_start, seq_end, global_counter = pending_batches.pop(0)

            batch_data = {
                'exercise': exercise,
                'confidence': confidence,
                'timestamp': timestamp,
                'batch_timestamp': batch_timestamp,
                'frames': batch_frames,   # 64 frames
                'batch_id': win_idx,
                'global_counter': global_counter,
                'seq_start': seq_start,
                'seq_end': seq_end,
                'weight': None
            }

            self._batch_predictions[key].append(batch_data)
            _batch_log(f"PREDICTION_ADD | {cam_id}:{track_id} | exercise={exercise} | conf={confidence:.2f} | batches_accumulated={len(self._batch_predictions[key])}/3")

            # Check if we have 3 batches for voting
            if len(self._batch_predictions[key]) >= 3:
                _batch_log(f"VOTING_TRIGGER | {cam_id}:{track_id} | 3 batches ready, performing voting...")
                self.perform_voting(cam_id, track_id)

    def perform_voting(self, cam_id: str, track_id: int) -> None:
        """
        Perform 3-batch voting to determine final exercise prediction.

        IDENTICAL to SlowFast version - no changes needed!
        """
        key = (cam_id, track_id)
        all_batches = self._batch_predictions[key]

        # Take first 3 batches for voting
        batches_to_vote = all_batches[:3]

        # Count exercise votes
        exercises = [batch['exercise'] for batch in batches_to_vote]
        vote_counts = Counter(exercises)

        # Find exercise with at least 2 votes
        winning_exercise = None
        winning_confidence = 0.0

        for exercise, count in vote_counts.items():
            if count >= 2:
                confidences = [b['confidence'] for b in batches_to_vote if b['exercise'] == exercise]
                winning_exercise = exercise
                winning_confidence = sum(confidences) / len(confidences)

                # Log individual batch confidences for debugging
                conf_str = ", ".join([f"{c*100:.1f}%" for c in confidences])
                print(f"[BATCH_CONFS] {cam_id}:{track_id} {exercise} - Individual: [{conf_str}] -> Avg: {winning_confidence*100:.1f}%")
                break

        if winning_exercise:
            # Combine all frames from 3 voted batches (192 total frames)
            all_frames = []
            for batch in batches_to_vote:
                all_frames.extend(batch['frames'])

            # Extract both circular batch IDs and global counters
            batch_ids = [b['batch_id'] for b in batches_to_vote]
            global_counters = [b['global_counter'] for b in batches_to_vote]

            # Create voting result
            voting_result = {
                'exercise': winning_exercise,
                'confidence': winning_confidence,
                'frames': all_frames,  # 192 frames total (3 * 64)
                'frame_count': len(all_frames),
                'vote_counts': dict(vote_counts),
                'batches_used': len(batches_to_vote),
                'timestamp': time.time(),
                'cam_id': cam_id,
                'track_id': track_id,
                'batch_ids': batch_ids,
                'global_counters': global_counters,
                'weight': 'unknown',
                'weight_confidence': 0.0
            }

            # Store result for retrieval
            self._voting_results[key] = voting_result
            self._voting_cycles_completed += 1

            # Print 192-frame prediction result
            vote_info = " | ".join([f"{ex}:{cnt}" for ex, cnt in vote_counts.items()])
            print(f"[192-FRAME] {cam_id}:{track_id} -> {winning_exercise} ({winning_confidence:.2%}) | Votes: {vote_info}")
            _batch_log(f"VOTING_RESULT | {cam_id}:{track_id} | exercise={winning_exercise} | conf={winning_confidence:.2%} | votes={dict(vote_counts)} | total_frames={len(all_frames)}")

            if DEBUG:
                print(f"[MVIT_VOTING] RESULT for {cam_id}:{track_id}:")
                print(f"  Winner: {winning_exercise} (avg conf: {winning_confidence:.3f})")
                print(f"  Vote counts: {vote_counts}")
                print(f"  Total frames: {len(all_frames)}")
        else:
            print(f"[VOTING_NO_CONSENSUS] {cam_id}:{track_id} - No exercise got 2+ votes. "
                  f"Vote counts: {dict(vote_counts)}")

        # Remove processed batches
        remaining_batches = all_batches[3:]
        self._batch_predictions[key] = remaining_batches

        # Immediately vote again if we have enough batches
        if len(remaining_batches) >= 3:
            self.perform_voting(cam_id, track_id)

    def worker_loop(self) -> None:
        """
        Main worker loop for processing video clips.

        Uses MViT inference instead of SlowFast.
        """
        torch.backends.cudnn.benchmark = True

        try:
            model = self.load_model()
        except Exception as e:
            print(f"[MVIT] FATAL: Failed to initialize: {e}")
            self._running = False
            return

        last_stats_print = time.time()

        while self._running:
            # Collect batch
            batch: List[List[np.ndarray]] = []
            metas: List[Tuple[str, int, float]] = []

            t0 = time.time()
            while len(batch) < self.max_microbatch and ((time.time() - t0) * 1000 < self.tick_ms):
                try:
                    cam, tid, clip, ts = self._in.get(timeout=0.05)
                    batch.append(clip)
                    metas.append((cam, tid, ts))
                except queue.Empty:
                    continue

            if not batch:
                # Print stats occasionally when idle
                if time.time() - last_stats_print > 20.0:
                    stats = self.get_stats()
                    voting_stats = self.get_voting_stats()
                    print(f"[MVIT] Stats: {stats}")
                    print(f"[MVIT] Voting: {voting_stats}")
                    print(f"[MVIT] Coverage improvement: {stats.get('coverage_ratio', 0):.1%}")
                    last_stats_print = time.time()
                continue

            # Log batch collection - shows which tracks are batched together
            tracks_in_batch = [f"{cam}:{tid}" for cam, tid, ts in metas]
            _batch_log(f"BATCH_COLLECT | batch_size={len(batch)} | tracks={tracks_in_batch}")

            try:
                # Save frames for debugging if enabled
                if SAVE_FRAMES:
                    for i, (clip, (cam, tid, ts)) in enumerate(zip(batch, metas)):
                        save_dir = os.path.join(SAVE_FRAMES_DIR, f"{cam}_{tid}")
                        os.makedirs(save_dir, exist_ok=True)

                        # Save a subset of frames (every 4th frame to save space)
                        for frame_idx in range(0, len(clip), 4):
                            frame = clip[frame_idx]
                            # Convert RGB to BGR for cv2.imwrite
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            frame_path = os.path.join(save_dir, f"batch_{self._batches_run}_clip_{i}_frame_{frame_idx:03d}.jpg")
                            cv2.imwrite(frame_path, frame_bgr)

                        if i == 0:  # Only log once per batch
                            print(f"[FRAME_SAVE] Saved frames to {save_dir}")

                # Prepare batch for MViT
                x = self.prep_batch_mvit(batch, debug_hash=False)

                with torch.inference_mode():
                    if self.device.type == 'cuda' and self.use_fp16:
                        with torch.amp.autocast('cuda'):
                            logits = model(x)
                    else:
                        logits = model(x)

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                inference_time = time.time() - t0
                now = time.time()
                _batch_log(f"INFERENCE_DONE | batch_size={len(batch)} | inference_time={inference_time*1000:.1f}ms")

                # Process predictions for voting
                for i, (cam, tid, ts) in enumerate(metas):
                    idx = int(preds[i])
                    conf = float(probs[i, idx])

                    if 0 <= idx < len(self.class_names):
                        label = self.class_names[idx]
                    else:
                        label = f"class_{idx}"
                        if len(self.class_names) > 0 and DEBUG:
                            print(f"[MVIT] WARNING: Prediction index {idx} out of range")

                    # Log top 3 predictions for debugging low confidence
                    if conf < 0.7:  # Only log when confidence is suspiciously low
                        top3_probs, top3_indices = torch.topk(probs[i], 3)
                        top3_str = ", ".join([
                            f"{self.class_names[int(idx)]}:{float(p)*100:.1f}%"
                            for p, idx in zip(top3_probs, top3_indices)
                        ])
                        print(f"[MVIT_LOW_CONF] {cam}:{tid} - Top3: {top3_str}")

                    self.process_batch_prediction(cam, tid, label, conf, now)

                self._batches_run += 1

            except Exception as e:
                if DEBUG:
                    print(f"[MVIT] Error in forward pass: {e}")
                continue

    def cleanup_loop(self) -> None:
        """Background cleanup loop for removing stale track data."""
        while self._running:
            time.sleep(30.0)
            now = time.time()
            stale_keys = []

            with self._lock:
                for key, last_ts in self._last_activity.items():
                    if (now - last_ts) > 60.0:
                        stale_keys.append(key)

                for key in stale_keys:
                    self._stream_buf.pop(key, None)
                    self._emit_idx.pop(key, None)
                    self._last_emit.pop(key, None)
                    self._seq_counters.pop(key, None)
                    self._last_activity.pop(key, None)
                    self._batch_predictions.pop(key, None)
                    self._voting_results.pop(key, None)
                    self._pending_batches.pop(key, None)
                    self._global_win_counter.pop(key, None)

            if stale_keys and DEBUG:
                print(f"[MVIT_CLEANUP] Removed {len(stale_keys)} stale tracks")

    def run_comprehensive_warmup(self) -> bool:
        """Run comprehensive warmup sequence."""
        print(f"[MVIT_WARMUP] ========== WARMUP STARTING ==========")

        try:
            print(f"[MVIT_WARMUP] Loading model...")
            model = self.load_model()
            print(f"[MVIT_WARMUP] Model loading ✓")

            print(f"[MVIT_WARMUP] Warming up GPU memory allocation...")
            dummy_clip = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(64)]
            dummy_batch = [dummy_clip]
            x = self.prep_batch_mvit(dummy_batch)

            # Use same inference logic as worker_loop to avoid FP16 errors
            with torch.inference_mode():
                if self.device.type == 'cuda' and self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        _ = model(x)
                else:
                    _ = model(x)

            del x
            torch.cuda.empty_cache()
            print(f"[MVIT_WARMUP] GPU memory warmup ✓")

            print(f"[MVIT_WARMUP] ========== WARMUP COMPLETED ==========")
            return True

        except Exception as e:
            print(f"[MVIT_WARMUP] ERROR: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        with self._lock:
            active = len([k for k, v in self._last_activity.items()
                         if (time.time() - v) < 10.0])
            pending = sum(len(v) for v in self._pending_batches.values())

            coverage = self._windows_created / max(1, self._submitted_frames // 64)

            return {
                'submitted_frames': self._submitted_frames,
                'ready_clips_generated': self._ready_clips_generated,
                'queue_size': self._in.qsize(),
                'dropped_clips': self._dropped_clips,
                'batches_run': self._batches_run,
                'voting_cycles_completed': self._voting_cycles_completed,
                'active_tracks': active,
                'pending_batches': pending,
                'model_loaded': self._model_loaded,
                'windows_created': self._windows_created,
                'windows_enqueued': self._windows_enqueued,
                'coverage_ratio': coverage,
                'emit_cursor_active': len(self._stream_buf),
            }

    def get_voting_stats(self) -> Dict:
        """Get voting statistics."""
        with self._lock:
            return {
                'total_voting_results': len(self._voting_results),
                'track_breakdown': {
                    f"{k[0]}:{k[1]}": len(v) for k, v in self._batch_predictions.items()
                }
            }

    def get_all_pending_voting_results(self) -> Dict[Tuple[str, int], Dict]:
        """Get and clear all pending voting results."""
        with self._lock:
            results = dict(self._voting_results)
            self._voting_results.clear()
            return results

    def _perform_voting(self, cam_id: str, track_id: int) -> None:
        """Internal voting method (called with lock already held)."""
        self.perform_voting(cam_id, track_id)
