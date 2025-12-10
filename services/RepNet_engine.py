import os
from typing import List, Optional, Dict
import torch
import torchvision.transforms as T
import numpy as np
from RepNet.repnet.model import RepNet


class RepNetEngine:
    """
    Lightweight in-memory RepNet runner.
    - Accepts a list of RGB frames (HxWx3, uint8).
    - Downsamples by stride, chunks into 64-frame clips, and returns a reps estimate.
    """
    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        default_stride: int = 2,
        input_size: int = 112
    ):
        """
        Initialize RepNet engine for repetition counting.

        Args:
            weights_path (str): Path to RepNet model weights file
            device (str): Device to run model on ('cuda' or 'cpu')
            default_stride (int): Default temporal downsampling stride
            input_size (int): Input image size for the model

        Raises:
            FileNotFoundError: If weights file is not found
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.default_stride = int(default_stride)
        self.input_size = int(input_size)

        # Model
        self.model = RepNet().to(self.device).eval()

        # Load weights
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"RepNet weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        self.model.load_state_dict(state_dict, strict=True)

        # Transform pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])

    @torch.no_grad()
    def infer_clip(
        self,
        clip_rgb: List[np.ndarray],
        stride: Optional[int] = None
    ) -> Dict[str, Optional[float]]:
        """
        Run RepNet on an in-memory clip to count repetitions.

        Processes a sequence of RGB frames through RepNet to detect and count
        repetitive motion patterns. Handles temporal downsampling and chunking
        into 64-frame segments for optimal processing.

        Args:
            clip_rgb (List[np.ndarray]): List of RGB frames (HxWx3, uint8).
                                        Can be longer than 64 frames; will be chunked.
            stride (Optional[int]): Temporal downsample factor. If None, uses default_stride.

        Returns:
            Dict[str, Optional[float]]: Dictionary containing:
                - "reps": Number of repetitions detected (int or None if insufficient frames)
                - "rep_conf": Confidence score for repetition detection (float)
                - "stride": Stride value used for processing (int)
        """
        s = int(stride or self.default_stride)
        if not clip_rgb or len(clip_rgb) < 64:
            return {"reps": None, "rep_conf": 0.0, "stride": s}

        # Downsample by stride and trim to a multiple of 64
        frames = clip_rgb[::s]
        n64 = (len(frames) // 64) * 64
        if n64 < 64:
            return {"reps": None, "rep_conf": 0.0, "stride": s}
        frames = frames[:n64]

        # Build tensor batches
        chunks = []
        for i in range(0, len(frames), 64):
            block = frames[i:i + 64]
            tensors = [self.transform(f) for f in block]
            x = torch.stack(tensors, dim=1)
            chunks.append(x)
        batch = torch.stack(chunks, dim=0).to(self.device)

        # Forward pass
        all_pl, all_ps = [], []
        for i in range(batch.shape[0]):
            pl, ps, _ = self.model(batch[i].unsqueeze(0))
            all_pl.append(pl.squeeze(0).detach().cpu())
            all_ps.append(ps.squeeze(0).detach().cpu())

        # Concatenate predictions and compute counts
        pl_cat = torch.cat(all_pl, dim=0)
        ps_cat = torch.cat(all_ps, dim=0)

        conf, period_length, period_count, periodicity_score = RepNet.get_counts(pl_cat, ps_cat, s)
        
        rep = float(period_count[-1]) if len(period_count) > 0 else 0.0
        n = round(rep)
        
        return {"reps": n, "rep_conf": float(conf), "stride": s}