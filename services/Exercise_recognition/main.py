"""
Main exercise recognition service (model-agnostic).

Orchestrates all components:
- Frame buffering
- Voting system
- Batch processing
- Model inference

Can swap models without changing this file.
"""
import threading
from typing import List, Optional, Dict
import numpy as np
from .models import VotingResult, TrackKey
from .frame_buffer import FrameBuffer
from .voting_system import VotingSystem
from .batch_processor import BatchProcessor
from .model_loader import SlowFastModelLoader
from .inference_loop import InferenceWorker


class ExerciseService:
    """
    Main exercise recognition service.

    Model-agnostic orchestrator that coordinates:
    - Frame buffering (create 64-frame windows)
    - Model inference (runs on windows)
    - Voting system (3-batch consensus)
    """

    def __init__(self,
                 class_names: List[str],
                 model_name: str = "slowfast_r50",
                 weights_path: Optional[str] = None,
                 device: str = "cuda",
                 use_fp16: bool = True,
                 confidence_threshold: float = 0.3,
                 max_batch_size: int = 12):
        """
        Initialize exercise recognition service.

        Args:
            class_names: List of exercise class names
            model_name: Model architecture name
            weights_path: Path to custom weights (optional)
            device: Device to use
            use_fp16: Use half precision
            confidence_threshold: Minimum confidence for predictions
            max_batch_size: Maximum clips per inference batch
        """
        print("[EXERCISE_SERVICE] Initializing exercise recognition service...")

        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.frame_buffer = FrameBuffer(window_size=64, stride=32)
        self.voting_system = VotingSystem(confidence_threshold=confidence_threshold)
        self.batch_processor = BatchProcessor(device=device, use_fp16=use_fp16)

        # Load model
        model_loader = SlowFastModelLoader(
            model_name=model_name,
            weights_path=weights_path,
            class_names=class_names,
            device=device,
            use_fp16=use_fp16
        )

        self.model = model_loader.load_model()

        # Run warmup
        success = model_loader.run_comprehensive_warmup(
            self.model,
            self.batch_processor
        )

        if not success:
            raise RuntimeError("Model warmup failed")

        # Initialize inference worker
        self.inference_worker = InferenceWorker(
            model=self.model,
            batch_processor=self.batch_processor,
            voting_system=self.voting_system,
            class_names=class_names,
            max_batch_size=max_batch_size
        )

        self.running = False

        print("[EXERCISE_SERVICE] Service initialized successfully")

    def start(self):
        """Start the service (starts inference worker)."""
        if self.running:
            return

        print("[EXERCISE_SERVICE] Starting service...")
        self.running = True
        self.inference_worker.start()

    def stop(self):
        """Stop the service."""
        if not self.running:
            return

        print("[EXERCISE_SERVICE] Stopping service...")
        self.running = False
        self.inference_worker.stop()

    def submit_crop(self, cam_id: str, track_id: int, frame: np.ndarray):
        """
        Submit a person crop for exercise recognition.

        This is the main API called by main.py for each detected person.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID
            frame: BGR crop of person

        Flow:
            1. Add to frame buffer
            2. Get completed windows
            3. Submit windows to inference worker
        """
        # Add frame to buffer
        windows = self.frame_buffer.add_frame(cam_id, track_id, frame)

        # Submit any completed windows for inference
        for window in windows:
            self.inference_worker.submit_window(window)

    def get_voting_results(self) -> Dict[TrackKey, VotingResult]:
        """
        Get completed voting results.

        Returns dictionary of {(cam_id, track_id): VotingResult}
        for all tracks that completed voting since last call.

        Called by main.py to get exercise detection results.
        """
        return self.voting_system.get_pending_results()

    def get_stats(self) -> dict:
        """Get comprehensive service statistics."""
        return {
            'frame_buffer': self.frame_buffer.get_stats(),
            'voting_system': self.voting_system.get_stats(),
            'inference_worker': self.inference_worker.get_stats(),
            'batch_processor': self.batch_processor.get_stats()
        }
