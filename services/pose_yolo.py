# services/pose_yolo.py
from typing import List, Dict
import numpy as np
import torch
from ultralytics import YOLO

class PoseYolo:
    """YOLO-Pose wrapper over a list of RGB frames."""
    def __init__(self, weights: str = "models/yolo11n-pose.pt",
                 imgsz: int = 256, conf: float = 0.25, iou: float = 0.45, device: str = None):
        """
        Initialize YOLO-Pose model for human pose estimation.

        Args:
            weights (str): Path to YOLO-Pose model weights file
            imgsz (int): Input image size for inference
            conf (float): Confidence threshold for detections
            iou (float): IoU threshold for non-maximum suppression
            device (str): Device to run inference on. If None, auto-selects CUDA or CPU
        """
        self.model = YOLO(weights)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    @torch.inference_mode()
    def infer_frames(self, frames_rgb: List[np.ndarray]) -> List[Dict]:
        """
        Run pose estimation on a list of RGB frames.

        Processes multiple RGB frames through YOLO-Pose model to detect human poses
        and extract keypoint coordinates with confidence scores.

        Args:
            frames_rgb (List[np.ndarray]): List of RGB frames (H, W, 3)

        Returns:
            List[Dict]: List of pose estimation results, one per frame. Each dict contains:
                - "keypoints": List of keypoints as [x, y, confidence] triplets
                - "score": Overall pose confidence score
        """
        if not frames_rgb:
            return []
        # YOLO expects BGR
        bgr = [f[..., ::-1].copy() for f in frames_rgb]
        results = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)
        out: List[Dict] = []
        for r in results:
            if not hasattr(r, "keypoints") or r.keypoints is None or len(r.keypoints) == 0:
                out.append({"keypoints": [], "score": 0.0})
                continue
            kp_xy = r.keypoints.xy           # (N, K, 2)
            kp_cf = getattr(r.keypoints, 'conf', None)
            if kp_cf is None:
                kp_cf = torch.ones(kp_xy.shape[:2], device=kp_xy.device)
            mean_conf = kp_cf.mean(dim=1)    # (N,)
            best = int(torch.argmax(mean_conf).item())
            xy = kp_xy[best].cpu().numpy()   # (K,2)
            cf = kp_cf[best].cpu().numpy()   # (K,)
            kps = [[float(x), float(y), float(c)] for (x, y), c in zip(xy, cf)]
            out.append({"keypoints": kps, "score": float(mean_conf[best].item())})
        return out