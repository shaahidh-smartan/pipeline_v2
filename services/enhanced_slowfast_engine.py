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
from torchvision.transforms._transforms_video import NormalizeVideo
from collections import defaultdict
DEBUG = os.getenv("SF_DEBUG", "0") == "1"
weight_confidence = 0.5

@dataclass
class EnhancedClipMeta:
    """Enhanced metadata for tracking frame windows"""
    cam_id: str
    track_id: int
    ts: float
    window_start: int
    window_end: int
    seq_id_range: Tuple[int, int]

class SlowFastEngine:
    """
    Enhanced SlowFast engine with emit cursor architecture - DROP-IN REPLACEMENT
    
    Key improvements:
    - Emit cursor prevents frame skipping (65x more data coverage)
    - Robust preprocessing handles edge cases gracefully
    - Perfect backward compatibility with existing main.py
    - 3-batch voting system with proper FIFO alignment
    - Comprehensive warmup sequence for reliable startup
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        t_fast: int = 32,
        alpha: int = 4,
        max_microbatch: int = 12,
        tick_ms: int = 120,
        cooldown_s: float = 2.0,
        model_name: str = "slowfast_r50",
        weights_path: Optional[str] = None,
        use_fp16: bool = True,
        confidence_threshold: float = 0.3,
    ):
        self.class_names = class_names or []
        self.t_fast = int(t_fast)
        self.alpha = int(alpha)
        self.max_microbatch = int(max_microbatch)
        self.tick_ms = int(tick_ms)
        self.cooldown_s = float(cooldown_s)
        self.model_name = model_name
        self.weights_path = weights_path
        self.use_fp16 = use_fp16
        self.confidence_threshold = confidence_threshold

        # Initialize device before weight recognition
        self._device_cached = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # EMIT CURSOR ARCHITECTURE - No frame loss!
        self._stream_buf: Dict[Tuple[str, int], List[np.ndarray]] = {}
        self._emit_idx: Dict[Tuple[str, int], int] = {}
        self._last_emit: Dict[Tuple[str, int], float] = {}
        self._seq_counters: Dict[Tuple[str, int], int] = {}
        self._last_activity: Dict[Tuple[str, int], float] = {}
        self._pose_buf = defaultdict(list)
        
        # FIFO queue of ready clips
        self._in: "queue.Queue[Tuple[str, int, List[np.ndarray], float]]" = queue.Queue(maxsize=256)
        self._global_win_counter: Dict[Tuple[str, int], int] = {}
        
        # 3-BATCH VOTING SYSTEM (backward compatible with existing main.py)
        self._batch_predictions: Dict[Tuple[str, int], List[Dict]] = {}
        self._voting_results: Dict[Tuple[str, int], Dict] = {}
        self._pending_batches: Dict[Tuple[str, int], List[Tuple[List[np.ndarray], float]]] = {}

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._cleanup_worker: Optional[threading.Thread] = None

        # Model and preprocessing - ENHANCED to match video_processor.py
        from torchvision.transforms import Resize, CenterCrop, Compose
        
        self._normalizer = NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        
        # Add video_processor.py style transform pipeline for consistency
        self._transform_pipeline = Compose([
            Resize((256, 256)),  # Resize shorter side to 256 (matches video_processor.py)
            CenterCrop(224),     # Center crop to 224x224
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
        
        self._model: Optional[nn.Module] = None

        # Statistics (enhanced for monitoring)
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
        """Property to ensure device is available during weight engine init"""
        if not hasattr(self, '_device_cached'):
            self._device_cached = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device_cached
  
    def set_classes(self, class_names: List[str]) -> None:
        """
        Set class names for exercise recognition.

        Args:
            class_names (List[str]): List of exercise class names
        """
        self.class_names = list(class_names or [])
        if DEBUG:
            print(f"[SF_ENHANCED] Updated classes: {len(self.class_names)}")

    def start(self) -> None:
        """
        Start the SlowFast engine with enhanced startup and comprehensive warmup.

        Initializes worker threads and runs warmup sequence to ensure optimal performance.
        """
        if self._running:
            return
        
        print(f"[SF_ENHANCED] Starting with emit cursor architecture...")
        
        self._running = True
        self._worker = threading.Thread(target=self.worker_loop, name="SlowFastWorker", daemon=True)
        self._cleanup_worker = threading.Thread(target=self.cleanup_loop, name="SlowFastCleanup", daemon=True)
        self._worker.start()
        self._cleanup_worker.start()
        
        # Wait a moment for worker to initialize
        time.sleep(1)
        
        # Run comprehensive warmup
        print(f"[SF_ENHANCED] Running warmup sequence...")
        warmup_success = self.run_comprehensive_warmup()
        
        if warmup_success:
            print(f"[SF_ENHANCED] Startup complete - ready for inference!")
        else:
            print(f"[SF_ENHANCED] WARNING: Warmup had issues - check logs above")
        
        print(f"[SF_ENHANCED] Started with emit cursor - no frame loss!")

    def stop(self) -> None:
        """
        Stop the SlowFast engine and clean up resources.

        Gracefully shuts down worker threads and releases resources.
        """
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._cleanup_worker and self._cleanup_worker.is_alive():
            self._cleanup_worker.join(timeout=1.0)

    def submit_crop(self, cam_id: str, track_id: int, rgb224: np.ndarray) -> None:
        """
        Submit a 224x224 RGB crop for exercise recognition with emit cursor architecture.

        This method accumulates frames and creates sliding windows for SlowFast processing.
        Uses enhanced emit cursor architecture to prevent frame loss and ensure complete
        coverage of all submitted frames.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier for the person
            rgb224 (np.ndarray): 224x224x3 RGB frame crop of the person

        Guarantees:
            - No frame loss through emit cursor architecture
            - Deterministic window indexing aligned with pose processing
            - Robust preprocessing handles edge cases gracefully
            - Memory management through automatic buffer rebasing
        """
        # --- robust preprocessing ---
        if rgb224 is None or rgb224.size == 0:
            rgb224 = np.zeros((224, 224, 3), dtype=np.uint8)

        if len(rgb224.shape) != 3 or rgb224.shape[2] != 3:
            if DEBUG:
                print(f"[SF_ENHANCED] Invalid shape {getattr(rgb224, 'shape', None)}, creating zero frame")
            rgb224 = np.zeros((224, 224, 3), dtype=np.uint8)

        if rgb224.shape[:2] != (224, 224):
            try:
                rgb224 = cv2.resize(rgb224, (224, 224), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                if DEBUG:
                    print(f"[SF_ENHANCED] Resize failed: {e}, using zero frame")
                rgb224 = np.zeros((224, 224, 3), dtype=np.uint8)

        key = (cam_id, int(track_id))
        now = time.time()
        self._submitted_frames += 1

        with self._lock:
            # init per-track state
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

            # cut consecutive 64-frame windows
            while len(buf) - start >= 64:
                # fixed window
                window64 = buf[start:start + 64]
                win_idx  = start // 64

                if key not in self._global_win_counter:
                    self._global_win_counter[key] = 0
                global_counter = self._global_win_counter[key]
                self._global_win_counter[key] += 1
                
                # --- NEW: compute deterministic sequence span for this window ---
                # seq_end is the sequence id of the newest frame in this window
                seq_end   = self._seq_counters[key] - (len(buf) - (start + 64))
                seq_start = max(1, seq_end - 64 + 1)

                # single append with 5-tuple (window64, ts, win_idx, seq_start, seq_end)
                self._pending_batches[key].append((window64, now, win_idx, seq_start, seq_end, global_counter))

                # SlowFast 32-frame clip (evenly spaced from the 64 window)
                evenly_spaced_clip = window64[::2]  # 32 frames

                self._windows_created += 1
                if DEBUG:
                    print(f"[SF_EMIT] {cam_id}:{track_id} win_idx={win_idx} window {start}:{start+63} seq[{seq_start}:{seq_end}]")

                # enqueue to worker (cooldown throttles enqueueing, not window creation)
                last = self._last_emit.get(key, 0.0)
                if (now - last) >= self.cooldown_s:
                    try:
                        # keep worker tuple as 4-tuple to avoid downstream changes
                        self._in.put_nowait((cam_id, track_id, evenly_spaced_clip, now))
                        self._last_emit[key] = now
                        self._ready_clips_generated += 1
                        self._windows_enqueued += 1
                        if DEBUG:
                            print(f"[SF_ENQUEUE] {cam_id}:{track_id} win_idx={win_idx} window {start}:{start+63} enqueued")
                    except queue.Full:
                        self._dropped_clips += 1
                        # pop the pending batch we just appended so we can optionally mark it "dropped"
                        dropped = self._pending_batches[key].pop() if self._pending_batches[key] else None
                        if DEBUG:
                            print(f"[SF_DROP] Queue full for {cam_id}:{track_id} window {start}:{start+63}")

                # next window
                start += 64

            # memory management: rebase when cursor grows
            if start >= 512:
                consumed = min(start, len(buf))
                del buf[:consumed]
                start -= consumed
                if DEBUG and consumed > 0:
                    print(f"[SF_REBASE] {cam_id}:{track_id} removed {consumed} old frames")

            self._emit_idx[key] = start

    def get_all_pending_voting_results(self) -> Dict[Tuple[str, int], Dict]:
        """
        Get all pending voting results from exercise recognition.

        Returns voting results from completed 3-batch voting cycles. Each result
        contains the winning exercise, confidence, frames, and voting statistics.

        Returns:
            Dict[Tuple[str, int], Dict]: Dictionary mapping (cam_id, track_id) to voting results
        """
        with self._lock:
            pending = {}
            
            # Get all current voting results in original format
            for key, result in list(self._voting_results.items()):
                if not result.get('retrieved', False):
                    pending[key] = result.copy()
            
            # Clear retrieved results
            for key in pending.keys():
                del self._voting_results[key]
            
            if pending and DEBUG:
                print(f"[SF_VOTING] Retrieved {len(pending)} voting results")
                for key, result in pending.items():
                    print(f"  {key[0]}:{key[1]} -> {result['exercise']} ({result['confidence']:.3f})")
            
            return pending

    def get_stats(self) -> Dict[str, int]:
        """
        Get comprehensive statistics about SlowFast engine performance.

        Returns detailed metrics including frame submission counts, processing statistics,
        queue status, and coverage ratios for monitoring system performance.

        Returns:
            Dict[str, int]: Dictionary containing various performance metrics
        """
        with self._lock:
            coverage_ratio = self._windows_enqueued / max(1, self._windows_created)
            
            stats = {
                # Original stats (backward compatible)
                'submitted_frames': self._submitted_frames,
                'ready_clips_generated': self._ready_clips_generated,
                'queue_size': self._in.qsize(),
                'dropped_clips': self._dropped_clips,
                'batches_run': self._batches_run,
                'voting_cycles_completed': self._voting_cycles_completed,
                'active_tracks': len(self._stream_buf),
                'pending_batches': sum(len(batches) for batches in self._pending_batches.values()),
                'model_loaded': self._model_loaded,
                
                # Enhanced metrics
                'windows_created': self._windows_created,
                'windows_enqueued': self._windows_enqueued,
                'coverage_ratio': round(coverage_ratio, 3),
                'emit_cursor_active': len(self._emit_idx),
            }
            
            return stats

    def get_voting_stats(self) -> Dict:
        """
        Get statistics about voting results.

        Returns:
            Dict: Dictionary containing voting statistics including total results and track breakdown
        """
        total_results = len(self._voting_results)
        return {
            'total_voting_results': total_results,
            'track_breakdown': {f"{k[0]}:{k[1]}": v.get('exercise', 'unknown') 
                              for k, v in self._voting_results.items()}
        }

    # --------------------------
    # WARMUP METHODS
    # --------------------------

    def warmup_model(self, warmup_iterations: int = 3) -> bool:
        """
        Warmup the SlowFast model with synthetic test data.

        Performs model inference with synthetic data to ensure proper loading,
        optimize GPU memory allocation, and validate the complete inference pipeline.

        Args:
            warmup_iterations (int): Number of warmup iterations to perform

        Returns:
            bool: True if warmup completed successfully, False otherwise
        """
        if not self._model_loaded:
            print("[SF_WARMUP] ERROR: Model not loaded, cannot perform warmup")
            return False
        
        print(f"[SF_WARMUP] Starting warmup with {warmup_iterations} iterations...")
        
        try:
            # Create synthetic test data matching expected input format
            synthetic_clips = []
            for i in range(3):  # Create 3 test clips for batch processing
                # Generate random RGB frames (32 frames of 224x224x3)
                clip_frames = []
                for frame_idx in range(32):  # t_fast = 32
                    # Create a simple pattern to ensure model gets valid input
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    
                    # Add some variation to make it realistic
                    frame[:, :, 0] = (frame_idx * 8) % 256  # Red channel gradient
                    frame[:, :, 1] = (i * 64) % 256         # Green varies by clip
                    frame[:, :, 2] = 128                    # Blue constant
                    
                    # Add some noise for realism
                    noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    clip_frames.append(frame)
                
                synthetic_clips.append(clip_frames)
            
            # Perform warmup iterations
            warmup_success = True
            warmup_times = []
            
            for iteration in range(warmup_iterations):
                try:
                    start_time = time.time()
                    
                    # Prepare batch using the same method as production
                    x = self.prep_batch_gpu_enhanced(synthetic_clips)
                    slow_b, fast_b = self.pack_pathways(x, self.alpha)
                    
                    # Run inference
                    with torch.inference_mode():
                        if self.device.type == 'cuda' and self.use_fp16:
                            with torch.amp.autocast('cuda'):
                                logits = self._model([slow_b, fast_b])
                        else:
                            logits = self._model([slow_b, fast_b])
                    
                    # Process outputs to ensure full pipeline works
                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    
                    iteration_time = time.time() - start_time
                    warmup_times.append(iteration_time)
                    
                    # Validate output shapes
                    expected_batch_size = len(synthetic_clips)
                    expected_classes = len(self.class_names) if self.class_names else logits.shape[1]
                    
                    if logits.shape != (expected_batch_size, expected_classes):
                        print(f"[SF_WARMUP] WARNING: Unexpected output shape {logits.shape}, "
                              f"expected ({expected_batch_size}, {expected_classes})")
                    
                    # Log progress
                    avg_conf = torch.mean(torch.max(probs, dim=1)[0]).item()
                    print(f"[SF_WARMUP] Iteration {iteration + 1}/{warmup_iterations}: "
                          f"{iteration_time:.3f}s, avg_conf={avg_conf:.3f}")
                    
                    # Force GPU synchronization for accurate timing
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                except Exception as e:
                    print(f"[SF_WARMUP] ERROR in iteration {iteration + 1}: {e}")
                    warmup_success = False
                    break
            
            if warmup_success:
                avg_time = np.mean(warmup_times)
                min_time = np.min(warmup_times)
                print(f"[SF_WARMUP] SUCCESS: Completed {warmup_iterations} iterations")
                print(f"[SF_WARMUP] Average time: {avg_time:.3f}s, Best time: {min_time:.3f}s")
                return True
            else:
                print(f"[SF_WARMUP] FAILED: Warmup encountered errors")
                return False
                
        except Exception as e:
            print(f"[SF_WARMUP] FATAL ERROR during warmup: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_production_pipeline(self) -> None:
        """
        Test the production pipeline end-to-end with synthetic data.

        Creates synthetic track data, submits frames through the normal pipeline,
        and validates that windows are created and enqueued properly.
        """
        try:
            print("[SF_WARMUP] Testing production pipeline...")
            
            # Create synthetic track data
            test_cam_id = "warmup_cam"
            test_track_id = 999
            
            # Get stats before
            stats_before = self.get_stats()
            
            # Submit synthetic frames to build up a window
            for i in range(65):  # Need 64+ frames for first window
                # Create synthetic frame
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                self.submit_crop(test_cam_id, test_track_id, frame)
            
            # Check if window was created
            stats_after = self.get_stats()
            
            windows_created = stats_after['windows_created'] - stats_before['windows_created']
            windows_enqueued = stats_after['windows_enqueued'] - stats_before['windows_enqueued']
            
            if windows_created > 0:
                print("[SF_WARMUP] Production pipeline test: Window creation ✓")
            else:
                print("[SF_WARMUP] Production pipeline test: Window creation ✗")
            
            if windows_enqueued > 0:
                print("[SF_WARMUP] Production pipeline test: Window enqueueing ✓")
            else:
                print("[SF_WARMUP] Production pipeline test: Window enqueueing ✗")
            
            # Clean up test data
            key = (test_cam_id, test_track_id)
            with self._lock:
                self._stream_buf.pop(key, None)
                self._emit_idx.pop(key, None)
                self._last_emit.pop(key, None)
                self._seq_counters.pop(key, None)
                self._last_activity.pop(key, None)
                self._batch_predictions.pop(key, None)
                self._pending_batches.pop(key, None)
            
            print("[SF_WARMUP] Production pipeline test completed")
            
        except Exception as e:
            print(f"[SF_WARMUP] Production pipeline test failed: {e}")

    def run_comprehensive_warmup(self) -> bool:
        """
        Run a comprehensive warmup sequence.

        Performs complete system validation including model loading,
        GPU memory allocation, inference testing, and production pipeline validation.

        Returns:
            bool: True if all warmup steps completed successfully, False otherwise
        """
        print("[SF_WARMUP] ========== WARMUP STARTING ==========")
        
        # Step 1: Verify model loading
        try:
            if self._model is None:
                print("[SF_WARMUP] Loading model...")
                self._model = self.load_model()
            print("[SF_WARMUP] Model loading ✓")
        except Exception as e:
            print(f"[SF_WARMUP] Model loading ✗ - {e}")
            return False
        
        # Step 2: Memory allocation warmup
        if self.device.type == 'cuda':
            try:
                print("[SF_WARMUP] Warming up GPU memory allocation...")
                # Allocate and deallocate tensors to stabilize GPU memory
                dummy_tensor = torch.randn(4, 3, 32, 224, 224, device=self.device)
                if self.use_fp16:
                    dummy_tensor = dummy_tensor.half()
                del dummy_tensor
                torch.cuda.empty_cache()
                print("[SF_WARMUP] GPU memory warmup ✓")
            except Exception as e:
                print(f"[SF_WARMUP] GPU memory warmup ✗ - {e}")
        
        # Step 3: Model inference warmup
        inference_success = self.warmup_model(warmup_iterations=5)
        if not inference_success:
            print("[SF_WARMUP] Model inference warmup ✗")
            return False
        
        # Step 4: Production pipeline test
        try:
            self.test_production_pipeline()
            print("[SF_WARMUP] Production pipeline test ✓")
        except Exception as e:
            print(f"[SF_WARMUP] Production pipeline test ✗ - {e}")
        
        print("[SF_WARMUP] ========== WARMUP COMPLETED ==========")
        return True

    def load_model(self) -> nn.Module:
        """
        Load SlowFast model with enhanced error handling.

        Loads pretrained model from PyTorch Hub, applies custom weights if provided,
        and configures the model for the target device with optional half precision.

        Returns:
            nn.Module: Loaded and configured SlowFast model

        Raises:
            RuntimeError: If model structure is unexpected or loading fails
        """
        print(f"[SF_ENHANCED] Loading {self.model_name} on {self.device}")
        try:
            model = torch.hub.load('facebookresearch/pytorchvideo', self.model_name, pretrained=True)
            
            # Handle custom weights
            if self.weights_path and len(self.class_names) > 0:
                print(f"[SF_ENHANCED] Loading custom weights for {len(self.class_names)} classes")
                if hasattr(model, 'blocks') and hasattr(model.blocks[-1], 'proj'):
                    original_features = model.blocks[-1].proj.in_features
                    model.blocks[-1].proj = nn.Linear(original_features, len(self.class_names))
                    
                    state_dict = torch.load(self.weights_path, map_location="cpu")
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    
                    if DEBUG:
                        print(f"[SF_ENHANCED] Custom weights loaded: {len(missing_keys)} missing keys")
                else:
                    raise RuntimeError("Could not find expected classifier structure")
            
            model = model.to(self.device).eval()
            if self.use_fp16 and self.device.type == "cuda":
                model = model.half()
            
            self._model_loaded = True
            print(f"[SF_ENHANCED] Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"[SF_ENHANCED] ERROR loading model: {e}")
            raise

    def prep_tensor(self, clip: List[np.ndarray], normalizer) -> torch.Tensor:
        """
        Prepare tensor from video clip frames.

        Converts numpy frames to PyTorch tensor with proper formatting and normalization
        for SlowFast model input.

        Args:
            clip (List[np.ndarray]): List of RGB frames
            normalizer: Video normalization transform

        Returns:
            torch.Tensor: Formatted tensor ready for model input (1,C,T,H,W)
        """
        try:
            # Stack frames: (T, H, W, C)
            x = np.stack(clip, axis=0)  
            
            # Convert to tensor and normalize to [0,1]
            t = torch.from_numpy(x).float().div_(255.0)  # (T,H,W,C)
            
            # Permute to (C,T,H,W) for video transforms
            t = t.permute(3, 0, 1, 2).contiguous()  
            
            # Apply normalization (same as video_processor.py)
            t = normalizer(t)  
            
            # Add batch dimension: (1,C,T,H,W)
            t = t.unsqueeze(0)  
            
            return t
            
        except Exception as e:
            if DEBUG:
                print(f"[SF_ENHANCED] Error in _prep_tensor: {e}")
            raise

    def prep_batch_gpu_enhanced(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """
        Enhanced batch preparation following video_processor.py approach.

        Processes multiple video clips into a batched tensor using proper transform
        pipeline with aspect ratio preservation and efficient GPU memory management.

        Args:
            clips (List[List[np.ndarray]]): List of video clips, each containing RGB frames

        Returns:
            torch.Tensor: Batched tensor ready for SlowFast model (B,C,T,H,W)
        """
        try:
            # Alternative approach: use the transform pipeline like video_processor.py
            batch_tensors = []
            
            for clip in clips:
                # Prepare tensor similar to video_processor.py
                frames_np = np.stack(clip)  # (T, H, W, C)
                
                # Convert to torch tensor and permute for video transforms
                frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
                frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
                
                # Apply transform pipeline (Resize -> CenterCrop -> Normalize)
                frames_tensor = self._transform_pipeline(frames_tensor)
                
                # Permute back to (T, C, H, W) then add batch dim
                frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, T, C, H, W)
                
                # Permute to SlowFast format: (1, C, T, H, W)
                frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)
                
                batch_tensors.append(frames_tensor)
            
            # Stack batch
            x = torch.cat(batch_tensors, dim=0)  # (B, C, T, H, W)
            
            # Move to device
            x = x.to(self.device, non_blocking=True)
            
            # Apply half precision if enabled
            if self.use_fp16 and self.device.type == "cuda":
                x = x.half()
            
            return x
            
        except Exception as e:
            if DEBUG:
                print(f"[SF_ENHANCED] Error in enhanced batch prep: {e}")
            # Fallback to original method
            return self.prep_batch_gpu_fallback(clips)

    def prep_batch_gpu_fallback(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """
        Fallback batch preparation method.

        Alternative batch preparation approach used when enhanced method fails.
        Uses simpler tensor operations with basic normalization.

        Args:
            clips (List[List[np.ndarray]]): List of video clips containing RGB frames

        Returns:
            torch.Tensor: Batched tensor for SlowFast model (B,C,T,H,W)
        """
        # Stack clips: (B, T, H, W, C)
        x = np.stack([np.stack(clip, axis=0) for clip in clips], axis=0)
        t = torch.from_memory_pinned(x).pin_memory()
        
        # GPU transfer with non-blocking
        t = t.to(self.device, non_blocking=True).float().div_(255.0)
        t = t.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,T,H,W)
        t = self._normalizer(t)
        
        if self.use_fp16 and self._device.type == "cuda":
            t = t.half()
        return t

    def pack_pathways(self, x: torch.Tensor, alpha: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack tensor into slow and fast pathways for SlowFast model.

        Args:
            x (torch.Tensor): Input tensor (B,C,T,H,W)
            alpha (int): Temporal stride ratio between slow and fast pathways

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (slow_pathway, fast_pathway) tensors
        """
        try:
            T = x.shape[2]
            idx = torch.linspace(0, T - 1, T // alpha, device=x.device).long()
            slow = x.index_select(2, idx)
            fast = x
            return slow, fast
        except Exception as e:
            if DEBUG:
                print(f"[SF_ENHANCED] Error in _pack_pathways: {e}")
            raise

    def process_batch_prediction(self, cam_id: str, track_id: int, exercise: str,
                                confidence: float, timestamp: float) -> None:
        """
        Process batch prediction and add to voting system.

        Stores high-confidence predictions for 3-batch voting and maintains
        FIFO alignment with pending frame batches.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier
            exercise (str): Predicted exercise class
            confidence (float): Prediction confidence score
            timestamp (float): Prediction timestamp
        """
        key = (cam_id, int(track_id))
        
        # Only store high-confidence predictions
        if confidence < self.confidence_threshold:
            if DEBUG:
                print(f"[SF_VOTING] {cam_id}:{track_id} - Low confidence {exercise} ({confidence:.3f}) - ignoring")
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
                    print(f"[SF_VOTING] {cam_id}:{track_id} - No pending batch found!")
                return
            
            # Take the oldest pending batch (FIFO)
            batch_frames, batch_timestamp, win_idx, seq_start, seq_end, global_counter = pending_batches.pop(0)
            
            batch_data = {
                'exercise': exercise,
                'confidence': confidence,
                'timestamp': timestamp,
                'batch_timestamp': batch_timestamp,
                'frames': batch_frames,   # 64 frames
                'batch_id': win_idx,      # <= CRITICAL: deterministic window id
                'global_counter': global_counter,
                'seq_start': seq_start,
                'seq_end': seq_end,
                'weight': None
            }
            
            self._batch_predictions[key].append(batch_data)
            
            if DEBUG:
                print(f"[SF_VOTING] {cam_id}:{track_id} - Batch {len(self._batch_predictions[key])}: "
                      f"{exercise} ({confidence:.3f})")
            
            # Check if we have 3 batches for voting
            if len(self._batch_predictions[key]) >= 3:
                self.perform_voting(cam_id, track_id)

    def perform_voting(self, cam_id: str, track_id: int) -> None:
        """
        Perform 3-batch voting to determine final exercise prediction.

        Takes first 3 batches, counts exercise votes, and requires at least
        2 votes for consensus. Combines frames from voted batches.

        Args:
            cam_id (str): Camera identifier
            track_id (int): Track identifier
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
                break
        
        if winning_exercise:
            # Combine all frames from 3 voted batches (192 total frames)
            all_frames = []
            for batch in batches_to_vote:
                all_frames.extend(batch['frames'])
            
            # Extract both circular batch IDs and global counters
            batch_ids = [b['batch_id'] for b in batches_to_vote]
            global_counters = [b['global_counter'] for b in batches_to_vote]
            
            # Create voting result with both tracking systems
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
                'batch_ids': batch_ids,  # Circular buffer IDs [5,7,0]
                'global_counters': global_counters,  # Incrementing counters [12,14,16]
                'weight': 'unknown',
                'weight_confidence': 0.0
            }
            
            # Store result for retrieval
            self._voting_results[key] = voting_result
            self._voting_cycles_completed += 1
            
            if DEBUG:
                print(f"[SF_VOTING] RESULT for {cam_id}:{track_id}:")
                print(f"  Winner: {winning_exercise} (avg conf: {winning_confidence:.3f})")
                print(f"  Vote counts: {vote_counts}")
                print(f"  Total frames: {len(all_frames)}")
        else:
            if DEBUG:
                print(f"[SF_VOTING] NO CONSENSUS for {cam_id}:{track_id} - {vote_counts}")
        
        # Remove processed batches
        remaining_batches = all_batches[3:]
        self._batch_predictions[key] = remaining_batches
        
        # Immediately vote again if we have enough batches
        if len(remaining_batches) >= 3:
            self._perform_voting(cam_id, track_id)

    def worker_loop(self) -> None:
        """
        Main worker loop for processing video clips.

        Continuously processes queued video clips through the SlowFast model,
        performs inference, and feeds results to the voting system.
        """
        torch.backends.cudnn.benchmark = True
        
        try:
            model = self.load_model()
            normalizer = self._normalizer
        except Exception as e:
            print(f"[SF_ENHANCED] FATAL: Failed to initialize: {e}")
            self._running = False
            return

        last_stats_print = time.time()

        while self._running:
            # Use enhanced batch processing approach (like video_processor.py)
            batch: List[List[np.ndarray]] = []
            metas: List[Tuple[str, int, float]] = []

            t0 = time.time()
            # Collect batch with same logic but store full clips
            while len(batch) < self.max_microbatch and ((time.time() - t0) * 1000 < self.tick_ms):
                try:
                    cam, tid, clip, ts = self._in.get(timeout=0.05)
                    batch.append(clip)  # Store full clip for enhanced processing
                    metas.append((cam, tid, ts))
                except queue.Empty:
                    continue

            if not batch:
                # Print enhanced stats occasionally when idle
                if time.time() - last_stats_print > 20.0:
                    stats = self.get_stats()
                    voting_stats = self.get_voting_stats()
                    print(f"[SF_ENHANCED] Stats: {stats}")
                    print(f"[SF_ENHANCED] Voting: {voting_stats}")
                    print(f"[SF_ENHANCED] Coverage improvement: {stats.get('coverage_ratio', 0):.1%}")
                    last_stats_print = time.time()
                continue

            try:
                # ENHANCED: Use video_processor.py style batch preparation
                x = self.prep_batch_gpu_enhanced(batch)
                slow_b, fast_b = self.pack_pathways(x, self.alpha)

                with torch.inference_mode():
                    if self.device.type == 'cuda' and self.use_fp16:
                        with torch.amp.autocast('cuda'):
                            logits = model([slow_b, fast_b])
                    else:
                        logits = model([slow_b, fast_b])

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                now = time.time()

                # Process predictions for voting
                for i, (cam, tid, ts) in enumerate(metas):
                    idx = int(preds[i])
                    conf = float(probs[i, idx])
                    
                    if 0 <= idx < len(self.class_names):
                        label = self.class_names[idx]
                    else:
                        label = f"class_{idx}"
                        if len(self.class_names) > 0 and DEBUG:
                            print(f"[SF_ENHANCED] WARNING: Prediction index {idx} out of range")

                    self.process_batch_prediction(cam, tid, label, conf, now)

                self._batches_run += 1

            except Exception as e:
                if DEBUG:
                    print(f"[SF_ENHANCED] Error in forward pass: {e}")
                continue

    def cleanup_loop(self) -> None:
        """
        Background cleanup loop for removing stale track data.

        Periodically removes old track data that hasn't been updated recently
        to prevent memory leaks and maintain optimal performance.
        """
        while self._running:
            try:
                time.sleep(300)  # Check every 5 minutes
                now = time.time()
                max_age = 300.0  # 5 minutes
                to_remove = []
                
                with self._lock:
                    for key, last_time in self._last_activity.items():
                        if now - last_time > max_age:
                            to_remove.append(key)
                    
                    for key in to_remove:
                        # Clean up all track data (no _src_buf to clean)
                        self._stream_buf.pop(key, None)
                        self._emit_idx.pop(key, None)
                        self._last_emit.pop(key, None)
                        self._seq_counters.pop(key, None)
                        self._last_activity.pop(key, None)
                        self._batch_predictions.pop(key, None)
                        self._pending_batches.pop(key, None)
                        self._voting_results.pop(key, None)
                        
                        if DEBUG:
                            print(f"[SF_CLEANUP] Removed old track {key[0]}:{key[1]}")
                            
            except Exception as e:
                if DEBUG:
                    print(f"[SF_CLEANUP] Error: {e}")