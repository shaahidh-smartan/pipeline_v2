"""
Batch preprocessing for SlowFast inference.

Handles:
- Frame resizing and normalization
- Tensor conversion
- Slow/Fast pathway splitting
- Batch preparation
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class BatchProcessor:
    """
    Prepares video clips for SlowFast inference.

    Handles all preprocessing:
    - Resize to 256x256
    - Center crop to 224x224
    - Normalization
    - Tensor conversion
    - Slow/Fast pathway packing
    """

    def __init__(self, device: str = "cuda", use_fp16: bool = True):
        """
        Initialize batch processor.

        Args:
            device: Device to use ("cuda" or "cpu")
            use_fp16: Use half precision for speed
        """
        self.device = torch.device(device)
        self.use_fp16 = use_fp16

    def prepare_batch(self, clips: List[List[np.ndarray]],
                     t_fast: int = 32, alpha: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch of clips for inference.

        Args:
            clips: List of clips, each clip is list of 64 BGR frames
            t_fast: Number of fast pathway frames (32)
            alpha: Slow pathway reduction factor (4)

        Returns:
            Tuple of (slow_pathway_tensor, fast_pathway_tensor)
        """
        if not clips:
            raise ValueError("Empty clips list")

        # Process each clip
        processed_clips = []

        for clip_frames in clips:
            # Preprocess frames
            processed = self.preprocess_clip(clip_frames)

            if processed is None:
                # Fallback: create dummy clip
                processed = torch.zeros((3, 32, 224, 224), dtype=torch.float32)

            processed_clips.append(processed)

        # Stack into batch
        batch = torch.stack(processed_clips, dim=0)  # (B, C, T, H, W)

        # Move to device
        batch = batch.to(self.device)

        if self.use_fp16 and self.device.type == "cuda":
            batch = batch.half()

        # Split into slow and fast pathways
        slow, fast = self.pack_pathways(batch, alpha=alpha)

        return slow, fast

    def preprocess_clip(self, frames: List[np.ndarray]) -> Optional[torch.Tensor]:
        """
        Preprocess single clip (64 frames → 32 frames tensor).

        Args:
            frames: List of 64 BGR frames

        Returns:
            Tensor of shape (C, T, H, W) or None if processing fails
        """
        try:
            if len(frames) < 64:
                return None

            # Sample 32 frames evenly from 64
            indices = np.linspace(0, 63, 32, dtype=int)
            sampled = [frames[i] for i in indices]

            # Process each frame
            processed_frames = []

            for frame in sampled:
                # Handle edge cases
                if frame is None or frame.size == 0:
                    # Use black frame
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)

                # Ensure correct shape
                if len(frame.shape) != 3:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)

                # Resize
                frame_resized = cv2.resize(frame, (256, 256),
                                          interpolation=cv2.INTER_AREA)

                # Center crop to 224x224
                h, w = frame_resized.shape[:2]
                start_h = (h - 224) // 2
                start_w = (w - 224) // 2
                frame_cropped = frame_resized[start_h:start_h+224,
                                             start_w:start_w+224]

                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)

                # To float and normalize
                frame_float = frame_rgb.astype(np.float32) / 255.0

                # Apply normalization
                mean = np.array([0.45, 0.45, 0.45])
                std = np.array([0.225, 0.225, 0.225])
                frame_normalized = (frame_float - mean) / std

                # Transpose to (C, H, W)
                frame_chw = np.transpose(frame_normalized, (2, 0, 1))

                processed_frames.append(frame_chw)

            # Stack to (C, T, H, W)
            clip_tensor = np.stack(processed_frames, axis=1)  # (C, T, H, W)

            return torch.from_numpy(clip_tensor).float()

        except Exception as e:
            print(f"[BATCH_PROC] Error preprocessing clip: {e}")
            return None

    def pack_pathways(self, x: torch.Tensor,
                     alpha: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split tensor into slow and fast pathways.

        Args:
            x: Input tensor (B, C, T, H, W)
            alpha: Temporal reduction for slow pathway

        Returns:
            Tuple of (slow_tensor, fast_tensor)
        """
        # Fast pathway: all frames
        fast = x

        # Slow pathway: subsample by alpha
        slow = x[:, :, ::alpha, :, :]

        return slow, fast

    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            'device': str(self.device),
            'use_fp16': self.use_fp16
        }
