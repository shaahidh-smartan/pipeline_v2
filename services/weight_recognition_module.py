# weight_recognition_module.py
"""
Weight recognition module for integration into SlowFast pipeline
Processes 8 evenly spaced frames from 64-frame windows
"""

import cv2
import json
import numpy as np
import torch
import timm
from ultralytics import YOLO
from collections import defaultdict

# ----------------------- CONFIG -----------------------
WEIGHTS_YOLO = "models/best_10.pt"
EMBEDDINGS_JSON = "models/new_embed2.json"
EMB_MODEL_NAME = "repvgg_a2"

# Detection params
IMGSZ = 640
CONF = 0.5
IOU = 0.45
MAX_DET = 6

# Recognition thresholds
DIST_ACCEPT = 0.95
DIST_MARGIN = 0.05

# Crop params
PAD_BASE = 0.05
PAD_MIN = 0.02
PAD_MAX = 0.08
MULTICROP_PAD_FACTORS = [0.6, 1.0, 1.35]

# Foreground masking
SAT_S = 0.28
SAT_V = 0.20
SAT_ERODE = 1
SAT_DILATE = 1


class WeightRecognitionEngine:
    """Weight recognition engine for SlowFast integration"""
    
    def __init__(self, device="cuda:0", use_fp16=True, use_prototypes=True):
        """
        Initialize Weight Recognition Engine.

        Args:
            device (str): Device to run models on (default: "cuda:0")
            use_fp16 (bool): Whether to use half precision for faster inference
            use_prototypes (bool): Whether to use prototype embeddings for recognition
        """
        self.device = device
        self.use_fp16 = use_fp16
        
        print(f"[WEIGHT_REC] Initializing weight recognition on {device}")
        
        # Load models
        self.yolo = YOLO(WEIGHTS_YOLO).to(device)
        self.embedder = self.build_embedder()
        
        # Load gallery
        E_np, M = self.load_embeddings(EMBEDDINGS_JSON)
        if use_prototypes:
            E_np, M = self.build_prototypes(E_np, M)
        
        self.gallery = torch.from_numpy(E_np).to(device, dtype=torch.float32)
        self.metadata = M
        
        print(f"[WEIGHT_REC] Loaded {len(self.metadata)} reference weights")
        print(f"[WEIGHT_DEBUG] YOLO model classes: {self.yolo.names}")
        # Warm up
        dummy = np.zeros((224, 224, 3), np.uint8)
        self.embed_crops([dummy])

    def detect_all_weights_frame(self, frame):
        """
        Detect all weights in a single frame using multi-crop recognition pipeline.

        Performs YOLO detection followed by embedding-based recognition with multiple
        crop padding factors for robust weight identification.

        Args:
            frame (numpy.ndarray): Input BGR frame

        Returns:
            List[dict]: List of weight detections, each containing:
                - 'bbox': Bounding box coordinates (x1, y1, x2, y2)
                - 'label': Weight class label
                - 'conf_det': YOLO detection confidence
                - 'conf_emb': Embedding similarity confidence
        """
        if frame is None or frame.size == 0:
            return []
        
        # Convert BGR to RGB for YOLO (same as new_rtsp.py)
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # YOLO prediction (using same settings as new_rtsp.py)
            res = self.yolo.predict(
                frame, 
                imgsz=IMGSZ,      # 832
                conf=CONF,        # Using new_rtsp.py conf instead of 0.1
                iou=IOU,          # 0.45
                verbose=False, 
                max_det=MAX_DET   # 12
            )[0]
            
            if res.boxes is None or len(res.boxes) == 0:
                return []
            
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf_scores = res.boxes.conf.cpu().numpy()
            
            # Build multi-crops for all detections (exactly like new_rtsp.py)
            all_crops = []
            all_det_idx = []
            boxes = []
            confs = []
            
            for di, ((x1, y1, x2, y2), cf) in enumerate(zip(xyxy, conf_scores)):
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                confs.append(float(cf))
                
                # Multi-crop with different padding factors (same as new_rtsp.py)
                for factor in MULTICROP_PAD_FACTORS:  # [0.6, 1.0, 1.35]
                    crop = self.make_crop(frame, x1, y1, x2, y2, PAD_BASE * factor)
                    if crop is not None:
                        crop = self.apply_mask(crop)  # Apply saturation mask
                        all_crops.append(crop)
                        all_det_idx.append(di)
            
            if not all_crops:
                return []
            
            # Embed all crops (using existing method)
            Q = self.embed_crops(all_crops)
            if Q.size == 0:
                return []
            
            # Search gallery for all embeddings (using existing method)
            weight_detections = []
            
            # Choose best crop per detection (same logic as new_rtsp.py)
            per_det = {}
            for i in range(Q.shape[0]):
                di = all_det_idx[i]
                
                # Search gallery for this embedding
                label, dist, margin = self.search_gallery(Q[i:i+1])  # Single embedding
                
                # Check thresholds (same as new_rtsp.py logic)
                if dist <= DIST_ACCEPT and margin >= DIST_MARGIN:
                    # Better detection logic from new_rtsp.py
                    cur = per_det.get(di)
                    better = (cur is None) or (dist < cur["d"] - 1e-6) or (abs(dist - cur["d"]) < 1e-6 and margin > cur["m"])
                    
                    if better:
                        per_det[di] = {
                            "d": dist, 
                            "m": margin, 
                            "label": label, 
                            "crop": all_crops[i]
                        }
            
            # Build final detection results (same format as monkey patch expected)
            for di, (x1, y1, x2, y2) in enumerate(boxes):
                cf = confs[di]
                info = per_det.get(di)
                
                if info is not None:
                    dist = info["d"]
                    margin = info["m"]
                    label = info["label"]
                    
                    # Convert distance to confidence (same as new_rtsp.py)
                    conf_emb = 1.0 - (dist ** 2) / 2.0  # Cosine similarity
                    
                    weight_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': label,
                        'conf_det': cf,      # YOLO detection confidence
                        'conf_emb': conf_emb # Embedding similarity confidence
                    })
                    
                    print(f"[WEIGHT_FRAME] Detected {label} at ({x1},{y1},{x2},{y2}) "
                        f"det_conf={cf:.3f} emb_conf={conf_emb:.3f} dist={dist:.3f}")
            
            return weight_detections
            
        except Exception as e:
            print(f"[WEIGHT_FRAME] Error in detect_all_weights_frame: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    # def process_window(self, window64):
    #     """
    #     Process 64-frame window with temporal sampling for weight recognition.

    #     Processes odd frames (32 total) from a 64-frame window to identify
    #     the most consistent weight class through voting.

    #     Args:
    #         window64 (list): List of 64 RGB frames

    #     Returns:
    #         tuple: (weight_label, confidence, total_detections)
    #             - weight_label (str): Consensus weight class or "unknown"
    #             - confidence (float): Average confidence score
    #             - total_detections (int): Total detections across all frames
    #     """
    #     if window64 is None or len(window64) != 64:
    #         return "unknown", 0.0, 0
        
    #     # Sample all odd frames (indices 1, 3, 5, 7, ..., 63) = 32 frames
    #     indices = list(range(1, 64, 2))  # Every odd frame
    #     frames = [window64[i] for i in indices]
        
    #     print(f"[WEIGHT_DEBUG] Processing {len(frames)} odd frames from 64-frame window")
        
    #     # Process each frame and collect votes
    #     weight_votes = defaultdict(float)
    #     total_detections = 0
    #     detection_details = []
        
    #     for i, frame in enumerate(frames):
    #         label, conf, n_det = self.process_frame(frame)
    #         detection_details.append(f"Frame {indices[i]}: {n_det} dets")
    #         if label != "unknown":
    #             weight_votes[label] += conf
    #         total_detections += n_det
        
    #     print(f"[WEIGHT_DEBUG] Detection summary: {', '.join(detection_details)}")
    #     print(f"[WEIGHT_DEBUG] Total detections across {len(frames)} frames: {total_detections}")
        
    #     # Get consensus weight
    #     if weight_votes:
    #         best_weight = max(weight_votes.items(), key=lambda x: x[1])
    #         valid_frames = len([f for f in frames if self.process_frame(f)[0] != "unknown"])
    #         avg_conf = best_weight[1] / max(1, valid_frames)
    #         print(f"[WEIGHT_DEBUG] Consensus: {best_weight[0]} (conf: {avg_conf:.3f})")
    #         return best_weight[0], avg_conf, total_detections
        
    #     print(f"[WEIGHT_DEBUG] No weight consensus found")
    #     return "unknown", 0.0, total_detections
    
    def process_frame(self, frame):
        """
        Process single frame for weight detection and recognition.

        Performs YOLO detection, multi-crop extraction, embedding generation,
        and gallery search to identify weights in the frame.

        Args:
            frame (numpy.ndarray): Input BGR frame

        Returns:
            tuple: (weight_label, confidence, num_detections)
                - weight_label (str): Best weight class found or "unknown"
                - confidence (float): Recognition confidence score
                - num_detections (int): Number of YOLO detections
        """
        if frame is None or frame.size == 0:
            return "unknown", 0.0, 0
        
        
        # CRITICAL DEBUG: Log frame properties
        print(f"[WEIGHT_FRAME] Input shape: {frame.shape}, dtype: {frame.dtype}, "
            f"min: {frame.min()}, max: {frame.max()}, mean: {frame.mean():.1f}")
        
        # Convert BGR to RGB
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CRITICAL DEBUG: Save a sample frame for manual inspection
        import os
        debug_dir = "weight_debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Save every 10th frame for manual inspection
        frame_count = getattr(self, '_debug_frame_count', 0)
        self._debug_frame_count = frame_count + 1
        
        if frame_count % 10 == 0:
            debug_path = f"{debug_dir}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(debug_path, frame)  # Save original BGR
            print(f"[WEIGHT_DEBUG] Saved debug frame: {debug_path}")
        
        # YOLO prediction with detailed logging
        try:
            print(f"[WEIGHT_YOLO] Running prediction on {frame.shape} RGB frame...")
            res = self.yolo.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, 
                                verbose=False, max_det=MAX_DET)[0]
            
            # Enhanced detection logging
            n_boxes = 0 if res.boxes is None else len(res.boxes)
            print(f"[WEIGHT_DET] YOLO returned {n_boxes} detections (conf_thresh={CONF}, iou={IOU})")
            
            if res.boxes is not None and len(res.boxes) > 0:
                # Log detection details
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf_scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else [0] * len(xyxy)
                
                for j, ((x1, y1, x2, y2), conf_score, cls) in enumerate(zip(xyxy, conf_scores, classes)):
                    box_w, box_h = x2 - x1, y2 - y1
                    class_name = self.yolo.names.get(int(cls), f"class_{int(cls)}")
                    print(f"[WEIGHT_DET] Box {j}: {class_name} conf={conf_score:.3f} "
                        f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) size={box_w:.0f}x{box_h:.0f}")
            else:
                print(f"[WEIGHT_DET] Zero detections - possible causes:")
                print(f"  - Confidence threshold too high (current: {CONF})")
                print(f"  - No weights visible in this specific frame")
                print(f"  - Model expects different input format")
                print(f"  - Frame content doesn't match training data")
                
        except Exception as e:
            print(f"[WEIGHT_ERROR] YOLO prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0, 0
        
        if res.boxes is None or len(res.boxes) == 0:
            return "unknown", 0.0, 0
        
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        
        # Process each detection
        best_label = "unknown"
        best_conf = 0.0
        
        for (x1, y1, x2, y2), cf in zip(xyxy, conf):
            # Get crops with different paddings
            crops = []
            for factor in MULTICROP_PAD_FACTORS:
                crop = self.make_crop(frame, x1, y1, x2, y2, PAD_BASE * factor)
                if crop is not None:
                    crop = self.apply_mask(crop)
                    crops.append(crop)
            
            if crops:
                # Embed crops
                Q = self.embed_crops(crops)
                if Q.size > 0:
                    # Find best match
                    label, dist, margin = self.search_gallery(Q)
                    
                    # Check thresholds
                    if dist <= DIST_ACCEPT and margin >= DIST_MARGIN:
                        # Convert distance to confidence
                        conf_score = 1.0 - (dist ** 2) / 2.0  # Cosine similarity
                        if conf_score > best_conf:
                            best_label = label
                            best_conf = conf_score
                            print(f"[WEIGHT_MATCH] Found weight: {label} (dist={dist:.3f}, conf={conf_score:.3f})")
        
        return best_label, best_conf, len(xyxy)
    
    def build_embedder(self):
        """
        Build RepVGG embedding model for feature extraction.

        Creates and configures a RepVGG model for generating embeddings
        from weight crop images.

        Returns:
            torch.nn.Module: Configured RepVGG model
        """
        m = timm.create_model(EMB_MODEL_NAME, pretrained=True, num_classes=0, 
                            global_pool="avg").to(self.device).eval()
        if self.use_fp16 and "cuda" in self.device:
            m.half()
        return m
    
    def load_embeddings(self, path):
        """
        Load reference embeddings from JSON file.

        Loads pre-computed embeddings and metadata for weight classes
        from a JSON file, normalizing embeddings for cosine similarity.

        Args:
            path (str): Path to JSON file containing embeddings

        Returns:
            tuple: (embeddings_array, metadata_list)
                - embeddings_array (numpy.ndarray): Normalized embeddings matrix
                - metadata_list (list): List of metadata dictionaries
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        embs, meta = [], []
        label_keys = ["dumbbell", "dummbbell", "video", "label", "id", "name"]
        
        for rec in data:
            e = np.asarray(rec.get("embedding", None), np.float32)
            if e.ndim != 1 or not np.all(np.isfinite(e)):
                continue
            e = e / (np.linalg.norm(e) + 1e-8)
            embs.append(e)
            
            # Find label
            label = "unknown"
            for k in label_keys:
                if k in rec:
                    label = str(rec[k]).replace(".mp4", "")
                    break
            meta.append({"video": label})
        
        return np.stack(embs, 0), meta
    
    def build_prototypes(self, E, M):
        """
        Build prototype embeddings by averaging embeddings per class.

        Creates representative embeddings for each weight class by averaging
        all available embeddings for that class.

        Args:
            E (numpy.ndarray): Array of embeddings
            M (list): List of metadata dictionaries

        Returns:
            tuple: (prototype_embeddings, prototype_metadata)
                - prototype_embeddings (numpy.ndarray): Averaged embeddings per class
                - prototype_metadata (list): Metadata for each prototype
        """
        buckets = defaultdict(list)
        
        for e, m in zip(E, M):
            buckets[m["video"]].append(e)
        
        keys, embs = [], []
        for k, vs in buckets.items():
            keys.append(k)
            m = np.mean(np.stack(vs, 0), 0)
            m = m / (np.linalg.norm(m) + 1e-8)
            embs.append(m)
        
        return np.stack(embs, 0), [{"video": k} for k in keys]
    
    @torch.no_grad()
    def embed_crops(self, crops_bgr):
        """
        Generate embeddings for crop images.

        Processes crop images through the embedding model to generate
        normalized feature vectors for similarity comparison.

        Args:
            crops_bgr (list): List of BGR crop images

        Returns:
            numpy.ndarray: Normalized embeddings matrix (N, feature_dim)
        """
        if not crops_bgr:
            return np.zeros((0, 0), np.float32)
        
        batch = []
        for c in crops_bgr:
            img = cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img[..., 0] = (img[..., 0] - 0.485) / 0.229
            img[..., 1] = (img[..., 1] - 0.456) / 0.224
            img[..., 2] = (img[..., 2] - 0.406) / 0.225
            batch.append(np.transpose(img, (2, 0, 1)))
        
        x = np.stack(batch, 0)
        dtype = torch.float16 if (self.use_fp16 and "cuda" in self.device) else torch.float32
        x_t = torch.from_numpy(x).to(self.device, dtype=dtype, non_blocking=True)
        z = self.embedder(x_t)
        z = torch.nn.functional.normalize(z, dim=1)
        return z.float().cpu().numpy()
    
    @torch.no_grad()
    def search_gallery(self, Q):
        """
        Search gallery for best matching weight class.

        Computes cosine similarities between query embeddings and gallery
        embeddings to find the best match with margin calculation.

        Args:
            Q (numpy.ndarray): Query embeddings matrix

        Returns:
            tuple: (label, distance, margin)
                - label (str): Best matching weight class
                - distance (float): Euclidean distance to best match
                - margin (float): Difference between best and second-best distances
        """
        if Q.shape[0] == 0:
            return "unknown", 999.0, 0.0
        
        Q_t = torch.from_numpy(Q).to(self.device, dtype=torch.float32, non_blocking=True)
        sims = Q_t @ self.gallery.T
        dist2 = 2.0 - 2.0 * sims
        
        # Get top-2 for margin calculation
        d2vals, idxs = torch.topk(dist2, k=min(2, dist2.shape[1]), dim=1, largest=False)
        
        # Find best across all crops
        best_idx = 0
        best_dist = 999.0
        best_margin = 0.0
        
        for i in range(Q.shape[0]):
            d1 = torch.sqrt(d2vals[i, 0]).item()
            d2 = torch.sqrt(d2vals[i, 1]).item() if d2vals.shape[1] > 1 else d1 + 1.0
            margin = d2 - d1
            
            if d1 < best_dist or (abs(d1 - best_dist) < 1e-6 and margin > best_margin):
                best_dist = d1
                best_margin = margin
                best_idx = idxs[i, 0].item()
        
        label = self.metadata[best_idx]["video"]
        return label, best_dist, best_margin
    
    def make_crop(self, frame, x1, y1, x2, y2, pad):
        """
        Create padded rectangular crop from detection bounding box.

        Extracts a crop from the frame with padding around the detection
        bounding box, ensuring crop stays within frame boundaries.

        Args:
            frame (numpy.ndarray): Input frame
            x1, y1, x2, y2 (float): Bounding box coordinates
            pad (float): Padding factor relative to box size

        Returns:
            numpy.ndarray or None: Cropped image or None if invalid
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
    
    def apply_mask(self, bgr):
        """
        Apply saturation-based foreground mask to emphasize weights.

        Uses HSV color space to create a mask that highlights saturated regions
        (likely weights) and grays out the background for better recognition.

        Args:
            bgr (numpy.ndarray): Input BGR image

        Returns:
            numpy.ndarray: Masked image with background regions grayed out
        """
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


# Integration function for enhanced_slowfast_engine.py
def integrate_weight_recognition(engine_instance=None):
    """
    Integration function for weight recognition with SlowFast engine.

    Provides a factory function to create or return a WeightRecognitionEngine instance
    for integration with the enhanced SlowFast engine pipeline.

    Args:
        engine_instance (WeightRecognitionEngine, optional): Existing engine instance

    Returns:
        WeightRecognitionEngine: Weight recognition engine instance

    Usage:
        Add this to enhanced_slowfast_engine.py in the submit_crop method
        after creating window64:

        # Weight recognition integration
        if self.weight_engine is not None:
            weight_label, weight_conf, n_det = self.weight_engine.process_window(window64)
            if weight_conf > 0.5:  # Threshold for valid weight
                print(f"[WEIGHT] {cam_id}:{track_id} - {weight_label} ({weight_conf:.3f})")
                # Store weight result with the window
                # You can add this to your tracking dictionaries
    """
    if engine_instance is None:
        return WeightRecognitionEngine()
    return engine_instance