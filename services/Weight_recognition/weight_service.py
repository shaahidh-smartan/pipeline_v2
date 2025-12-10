"""
Main weight detection and recognition service.

Orchestrates detector and recognizer for complete weight recognition pipeline.
"""
import numpy as np
from .weight_detector import WeightDetector
from .weight_recognizer import WeightRecognizer
from .config import PAD_BASE, MULTICROP_PAD_FACTORS


class WeightService:
    """
    Main weight recognition service.

    Combines YOLO detection and embedding recognition
    for complete weight identification pipeline.
    """

    def __init__(self, device="cuda:0", use_fp16=True, use_prototypes=True):
        """
        Initialize weight service.

        Args:
            device: Device to run models on
            use_fp16: Use half precision
            use_prototypes: Use prototype embeddings
        """
        self.device = device

        print(f"[WEIGHT_SERVICE] Initializing on {device}")

        # Initialize components
        self.detector = WeightDetector(device=device)
        self.recognizer = WeightRecognizer(
            device=device,
            use_fp16=use_fp16,
            use_prototypes=use_prototypes
        )

        print(f"[WEIGHT_SERVICE] Service initialized successfully")

    def detect_and_recognize(self, frame):
        """
        Full pipeline: detect and recognize weights in frame.

        Args:
            frame: BGR input frame

        Returns:
            List of weight detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - label: Weight class label
                - conf_det: Detection confidence
                - conf_emb: Recognition confidence
        """
        if frame is None or frame.size == 0:
            return []

        # Step 1: Detect weights with YOLO
        detections = self.detector.detect(frame)

        if not detections:
            return []

        # Step 2: Recognize each detection
        results = []

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            conf_det = det['conf']

            # Create multi-crops with different padding
            crops = []
            for factor in MULTICROP_PAD_FACTORS:
                crop = self.detector.make_crop(frame, x1, y1, x2, y2, PAD_BASE * factor)
                if crop is not None:
                    # Apply saturation mask
                    crop = self.detector.apply_saturation_mask(crop)
                    crops.append(crop)

            if not crops:
                continue

            # Recognize weight from crops
            label, conf_emb = self.recognizer.recognize(crops)

            if label != "unknown":
                results.append({
                    'bbox': bbox,
                    'label': label,
                    'conf_det': conf_det,
                    'conf_emb': conf_emb
                })

                print(f"[WEIGHT_SERVICE] Detected {label} at {bbox} "
                      f"(det={conf_det:.3f}, emb={conf_emb:.3f})")

        return results

    def process_frame(self, frame):
        """
        Process single frame for weight detection.

        Simplified interface that returns best weight found.

        Args:
            frame: BGR input frame

        Returns:
            (weight_label, confidence, num_detections)
        """
        results = self.detect_and_recognize(frame)

        if not results:
            return "unknown", 0.0, 0

        # Return best detection
        best = max(results, key=lambda x: x['conf_emb'])
        return best['label'], best['conf_emb'], len(results)

    def get_stats(self):
        """Get service statistics."""
        return {
            'device': self.device,
            'gallery_size': len(self.recognizer.metadata)
        }
