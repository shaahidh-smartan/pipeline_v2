import torch
from ultralytics import YOLO
import numpy as np
import threading


class PersonDetector:
    """Thread-safe YOLO-based person detection utility."""
    
    def __init__(self, model_name='yolo11m-pose.pt', confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        
        self.model = YOLO(model_name)
        
        # Configure thread safety for YOLO
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'eval'):
            self.model.model.eval()  # Set to evaluation mode for thread safety
        
        # Thread safety lock for inference calls
        self._inference_lock = threading.Lock()
        
        # Move to GPU if available with stream isolation
        if torch.cuda.is_available():
            self.device = 'cuda'
            # Force CUDA context creation and set memory management
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False  # Disable for deterministic behavior
            torch.backends.cudnn.deterministic = True  # Ensure deterministic results
        else:
            self.device = 'cpu'
        
    
    def detect_persons(self, frame):
        """
        Detect persons in frame using YOLO with thread-safe inference.

        Returns:
            person_boxes: List of dict with bbox, confidence, and keypoints
            person_boxes_track: List for tracking [x1, y1, x2, y2, confidence]
        """
        try:
            # Thread-safe inference with explicit CUDA synchronization
            with self._inference_lock:
                # Ensure CUDA synchronization before inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                results = self.model(frame, verbose=False)

                # Ensure CUDA synchronization after inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            person_boxes = []
            person_boxes_track = []

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # Extract keypoints if available (YOLO pose model)
                keypoints_data = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints

                for i, box in enumerate(boxes):
                    # Ensure cls and conf exist and are indexable
                    if box.cls is None or box.conf is None:
                        continue

                    cls_val = int(box.cls.item()) if box.cls.numel() == 1 else int(box.cls[0].item())
                    conf_val = float(box.conf.item()) if box.conf.numel() == 1 else float(box.conf[0].item())

                    # Class 0 is person in COCO dataset
                    if cls_val == 0 and conf_val > self.confidence_threshold:
                        if box.xyxy is None or box.xyxy.numel() < 4:
                            continue

                        xyxy = box.xyxy[0] if box.xyxy.ndim > 1 else box.xyxy
                        x1, y1, x2, y2 = xyxy.cpu().numpy()

                        # Extract keypoints for this person [17, 3] -> (x, y, conf)
                        kps = None
                        if keypoints_data is not None:
                            try:
                                kps = keypoints_data.data[i].cpu().numpy()  # [17, 3]
                            except (IndexError, AttributeError):
                                kps = None

                        person_boxes.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf_val,
                            'keypoints': kps
                        })
                        person_boxes_track.append([int(x1), int(y1), int(x2), int(y2), conf_val])

            return person_boxes, person_boxes_track

        except Exception as e:
            return [], []

    
    def detect_persons_yolov5(self, frame):
        """Alternative YOLOv5 detection method for compatibility."""
        try:
            # Load YOLOv5 if not already loaded
            if not hasattr(self, 'yolov5_model'):
                self.yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                if torch.cuda.is_available():
                    self.yolov5_model = self.yolov5_model.cuda()
            
            results = self.yolov5_model(frame)
            person_boxes = []
            person_boxes_track = []
            
            # Handle different YOLOv5 result formats
            if hasattr(results, 'xyxy'):
                # Newer format
                detections = results.xyxy[0].cpu().numpy()
                for detection in detections:
                    # Class 0 is person in COCO dataset
                    if int(detection[5]) == 0 and float(detection[4]) > self.confidence_threshold:
                        x1, y1, x2, y2 = detection[:4]
                        confidence = float(detection[4])
                        
                        person_boxes.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence
                        })
                        person_boxes_track.append([int(x1), int(y1), int(x2), int(y2), confidence])
            
            return person_boxes, person_boxes_track
            
        except Exception as e:
            return [], []
    
    def filter_person_detections(self, person_boxes_track, min_area=100, max_aspect_ratio=2.0):
        """Filter person detections by size and aspect ratio."""
        filtered_detections = []
        
        for box in person_boxes_track:
            if len(box) >= 5:
                x1, y1, x2, y2, score = box[:5]
                w = x2 - x1
                h = y2 - y1
                
                # Calculate area and aspect ratio
                area = w * h
                aspect_ratio = w / h if h > 0 else float('inf')
                
                # Apply filters
                if area > min_area and aspect_ratio <= max_aspect_ratio and score > 0:
                    filtered_detections.append(box)
        
        return filtered_detections
    
    def get_person_crop(self, frame, bbox, padding=10):
        """Extract person crop from frame with optional padding."""
        try:
            x1, y1, x2, y2 = bbox[:4]
            
            # Add padding
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(frame_w, x2 + padding)
            y2 = min(frame_h, y2 + padding)
            
            # Extract crop
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
            
            return person_crop
            
        except Exception as e:
            return None