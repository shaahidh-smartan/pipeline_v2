"""
YOLO-based weight plate detection.

Handles detection of weight plates in frames using YOLO model.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from .config import WEIGHTS_YOLO, IMGSZ, CONF, IOU, MAX_DET


class WeightDetector:
    """
    YOLO-based weight plate detector.

    Uses YOLOv8 to detect weight plates in frames.
    """

    def __init__(self, model_path=WEIGHTS_YOLO, device="cuda:0"):
        """
        Initialize weight detector.

        Args:
            model_path: Path to YOLO model weights
            device: Device to run model on
        """
        self.device = device
        self.yolo = YOLO(model_path).to(device)

        print(f"[WEIGHT_DETECTOR] Initialized on {device}")
        print(f"[WEIGHT_DETECTOR] Model classes: {self.yolo.names}")

    def detect(self, frame):
        """
        Detect weight plates in frame.

        Args:
            frame: BGR input frame

        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - conf: Detection confidence
                - cls: Class ID
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # Run YOLO prediction
            results = self.yolo.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False,
                max_det=MAX_DET
            )[0]

            if results.boxes is None or len(results.boxes) == 0:
                return []

            # Extract detection data
            xyxy = results.boxes.xyxy.cpu().numpy()
            conf_scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else [0] * len(xyxy)

            # Package detections
            detections = []
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, conf_scores, classes):
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(conf),
                    'cls': int(cls)
                })

            print(f"[WEIGHT_DETECTOR] Found {len(detections)} detections")
            return detections

        except Exception as e:
            print(f"[WEIGHT_DETECTOR] Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def make_crop(self, frame, x1, y1, x2, y2, pad):
        """
        Create padded crop from bounding box.

        Args:
            frame: Input frame
            x1, y1, x2, y2: Bounding box coordinates
            pad: Padding factor relative to box size

        Returns:
            Cropped image or None if invalid
        """
        H, W = frame.shape[:2]
        w, h = (x2 - x1), (y2 - y1)

        if w <= 1 or h <= 1:
            return None

        # Apply padding
        dw, dh = w * pad, h * pad
        x1n = int(np.floor(max(0, x1 - dw)))
        y1n = int(np.floor(max(0, y1 - dh)))
        x2n = int(np.ceil(min(W - 1, x2 + dw)))
        y2n = int(np.ceil(min(H - 1, y2 + dh)))

        if x2n <= x1n or y2n <= y1n:
            return None

        return frame[y1n:y2n, x1n:x2n]

    def apply_saturation_mask(self, bgr):
        """
        Apply saturation-based foreground mask.

        Uses HSV color space to highlight saturated regions
        (likely weights) and gray out background.

        Args:
            bgr: Input BGR image

        Returns:
            Masked image with background grayed out
        """
        from .config import SAT_S, SAT_V, SAT_ERODE, SAT_DILATE

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        s, v = hsv[..., 1], hsv[..., 2]
        m = (s >= SAT_S) & (v >= SAT_V)
        m = m.astype(np.uint8) * 255

        if SAT_ERODE > 0:
            m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=SAT_ERODE)
        if SAT_DILATE > 0:
            m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=SAT_DILATE)

        out = bgr.copy()
        out[m == 0] = 127  # Gray out background
        return out
