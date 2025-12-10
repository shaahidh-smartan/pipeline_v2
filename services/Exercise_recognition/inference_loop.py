"""
Inference worker thread for exercise recognition.

Runs continuously:
- Collects clips from queue
- Runs inference
- Sends predictions to voting system
"""
import time
import queue
import threading
import torch
from typing import Optional, List
from .models import FrameWindow, BatchPrediction, TrackKey


class InferenceWorker:
    """
    Worker thread that runs model inference continuously.

    Flow:
    1. Collect clips from queue
    2. Batch clips together
    3. Run model inference
    4. Send predictions to voting system
    """

    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor,
                 voting_system,
                 class_names: List[str],
                 max_batch_size: int = 12,
                 tick_ms: int = 120):
        """
        Initialize inference worker.

        Args:
            model: Loaded SlowFast model
            batch_processor: BatchProcessor instance
            voting_system: VotingSystem instance
            class_names: List of exercise class names
            max_batch_size: Maximum clips per batch
            tick_ms: Milliseconds to wait for clips
        """
        self.model = model
        self.batch_processor = batch_processor
        self.voting_system = voting_system
        self.class_names = class_names
        self.max_batch_size = max_batch_size
        self.tick_ms = tick_ms

        # Queue for incoming windows
        self.clip_queue: queue.Queue = queue.Queue(maxsize=256)

        # Worker thread
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Statistics
        self.batches_processed = 0
        self.total_inference_time = 0.0

    def start(self):
        """Start the inference worker thread."""
        if self.running:
            return

        print("[INFERENCE_WORKER] Starting inference worker...")
        self.running = True
        self.worker_thread = threading.Thread(
            target=self.worker_loop,
            name="InferenceWorker",
            daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        """Stop the inference worker thread."""
        if not self.running:
            return

        print("[INFERENCE_WORKER] Stopping inference worker...")
        self.running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

    def submit_window(self, window: FrameWindow):
        """
        Submit a frame window for inference.

        Args:
            window: FrameWindow to process
        """
        try:
            self.clip_queue.put_nowait(window)
        except queue.Full:
            print(f"[INFERENCE_WORKER] Queue full, dropping window "
                  f"{window.cam_id}:{window.track_id}")

    def worker_loop(self):
        """Main worker loop - collects clips and runs inference."""
        print("[INFERENCE_WORKER] Worker loop started")

        while self.running:
            # Collect batch of clips
            windows = self.collect_batch()

            if not windows:
                time.sleep(0.01)
                continue

            # Run inference on batch
            self.process_batch(windows)

    def collect_batch(self) -> List[FrameWindow]:
        """
        Collect batch of windows from queue.

        Returns:
            List of FrameWindow objects (up to max_batch_size)
        """
        windows = []
        timeout = self.tick_ms / 1000.0  # Convert to seconds

        try:
            # Wait for first clip
            first = self.clip_queue.get(timeout=timeout)
            windows.append(first)

            # Collect additional clips (non-blocking)
            while len(windows) < self.max_batch_size:
                try:
                    window = self.clip_queue.get_nowait()
                    windows.append(window)
                except queue.Empty:
                    break

        except queue.Empty:
            pass

        return windows

    def process_batch(self, windows: List[FrameWindow]):
        """
        Process batch of windows through model.

        Args:
            windows: List of FrameWindow objects
        """
        try:
            # Prepare batch
            clips = [window.frames for window in windows]
            slow, fast = self.batch_processor.prepare_batch(clips)

            # Run inference
            with torch.no_grad():
                start = time.time()
                logits = self.model([slow, fast])
                inference_time = time.time() - start

            # Update statistics
            self.batches_processed += 1
            self.total_inference_time += inference_time

            # Process predictions
            probs = torch.softmax(logits, dim=1)

            for i, window in enumerate(windows):
                # Get top prediction
                prob = probs[i]
                top_prob, top_idx = torch.max(prob, dim=0)

                exercise = self.class_names[top_idx.item()]
                confidence = top_prob.item()

                # Create prediction
                prediction = BatchPrediction(
                    exercise=exercise,
                    confidence=confidence,
                    frames=window.frames,
                    batch_id=window.window_index,
                    timestamp=time.time()
                )

                # Send to voting system
                result = self.voting_system.add_prediction(
                    window.cam_id,
                    window.track_id,
                    prediction
                )

                if result:
                    print(f"[VOTING_COMPLETE] {window.cam_id}:{window.track_id} - "
                          f"{result.exercise} ({result.confidence:.3f})")

        except Exception as e:
            print(f"[INFERENCE_WORKER] Error processing batch: {e}")

    def get_stats(self) -> dict:
        """Get worker statistics."""
        avg_time = (self.total_inference_time / self.batches_processed
                   if self.batches_processed > 0 else 0)

        return {
            'batches_processed': self.batches_processed,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time_ms': avg_time * 1000,
            'queue_size': self.clip_queue.qsize()
        }
