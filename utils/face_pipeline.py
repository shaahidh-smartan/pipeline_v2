import cv2
import numpy as np
import sys
import os
import threading
import torch

# Add face registration module to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_registration_module'))

from face_detection import SCRFD
from face_alignment import AlignCrop  
from embedding_generator import ArcFace


class FacePipeline:
    """complete face detection, alignment, and embedding pipeline."""
    
    def __init__(self,
                 detection_model='models/scrfd_10g_bnkps.onnx',
                 embedding_model='models/mbf.onnx',
                 gpu_id=0):
        """Initialize face pipeline components with thread safety."""
        self.gpu_id = gpu_id
        
        # Thread safety locks for each component
        self._detector_lock = threading.Lock()
        self._embedder_lock = threading.Lock()
        self._aligner_lock = threading.Lock()  # CPU-based but still needs protection
        
        # Face detector
        print("Initializing thread-safe face detector...")
        self.detector = SCRFD(model_file=detection_model)
        self.detector.prepare(gpu_id)
        
        # Face alignment (CPU-based but still needs thread protection)
        print("Initializing thread-safe face alignment...")
        self.aligner = AlignCrop()
        
        # Face embedder
        print("Initializing thread-safe face embedder...")
        self.embedder = ArcFace(model_file=embedding_model)
        self.embedder.prepare(gpu_id)
        
        # Configure CUDA settings for thread safety
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False  # Disable for deterministic behavior
            torch.backends.cudnn.deterministic = True
        
        print("Thread-safe face pipeline initialized successfully")
        print(f"[INFO] Face pipeline using GPU {gpu_id} with thread-safe inference")
    
    def detect_faces(self, frame, thresh=0.5, input_size=(640, 640)):
        """Detect faces in frame with thread-safe inference."""
        try:
            with self._detector_lock:
                # Ensure CUDA synchronization before inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                bboxes, keypoints = self.detector.detect(frame, thresh=thresh, input_size=input_size)
                
                # Ensure CUDA synchronization after inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            return bboxes, keypoints
        except Exception as e:
            print(f"Error in face detection: {e}")
            return [], []
    
    def align_face(self, frame, keypoints, image_size=112):
        """Align and crop face using keypoints with thread-safe processing."""
        try:
            with self._aligner_lock:
                aligned_face, transform_matrix = self.aligner.align_and_crop(
                    frame, keypoints, image_size=image_size
                )
            return aligned_face, transform_matrix
        except Exception as e:
            print(f"Error in face alignment: {e}")
            return None, None
    
    def get_face_embedding(self, aligned_face):
        """Get embedding from aligned face with thread-safe inference."""
        try:
            with self._embedder_lock:
                # Ensure CUDA synchronization before inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                embedding = self.embedder.get_embedding(aligned_face)
                
                # Ensure CUDA synchronization after inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Normalize embedding
            normalized_embedding = embedding / np.linalg.norm(embedding)
            return normalized_embedding
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            return None
    
    def process_frame(self, frame, thresh=0.5, input_size=(640, 640), image_size=112):
        """
        Complete pipeline: detect -> align -> embed.
        
        Returns:
            List of dictionaries with:
            - bbox: face bounding box
            - keypoints: facial keypoints
            - aligned_face: aligned face crop
            - embedding: normalized face embedding
            - confidence: detection confidence
        """
        results = []
        
        try:
            # Step 1: Detect faces
            bboxes, keypoints_list = self.detect_faces(frame, thresh, input_size)
            
            if len(bboxes) == 0:
                return results
            
            # Step 2: Process each detected face
            for bbox, keypoints in zip(bboxes, keypoints_list):
                face_data = {
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'confidence': bbox[4],
                    'aligned_face': None,
                    'embedding': None
                }
                
                # Step 3: Align face
                aligned_face, transform_matrix = self.align_face(frame, keypoints, image_size)
                
                if aligned_face is not None:
                    face_data['aligned_face'] = aligned_face
                    
                    # Step 4: Get embedding
                    embedding = self.get_face_embedding(aligned_face)
                    
                    if embedding is not None:
                        face_data['embedding'] = embedding
                
                results.append(face_data)
            
            return results
            
        except Exception as e:
            print(f"Error in face pipeline processing: {e}")
            return []
    
    def process_single_face(self, frame, bbox=None, keypoints=None, thresh=0.5, image_size=112):
        """Process a single face region or detect the best face."""
        try:
            if bbox is None or keypoints is None:
                # Detect faces and use the best one
                bboxes, keypoints_list = self.detect_faces(frame, thresh)
                
                if len(bboxes) == 0:
                    return None
                
                # Use face with highest confidence
                best_idx = np.argmax([bbox[4] for bbox in bboxes])
                bbox = bboxes[best_idx]
                keypoints = keypoints_list[best_idx]
            
            # Align and embed
            aligned_face, _ = self.align_face(frame, keypoints, image_size)
            
            if aligned_face is not None:
                embedding = self.get_face_embedding(aligned_face)
                
                return {
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'aligned_face': aligned_face,
                    'embedding': embedding,
                    'confidence': bbox[4]
                }
            
            return None
            
        except Exception as e:
            print(f"Error processing single face: {e}")
            return None
    
    def batch_process_images(self, images, thresh=0.5, input_size=(640, 640), image_size=112):
        """Process multiple images in batch."""
        all_results = []
        
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            results = self.process_frame(image, thresh, input_size, image_size)
            all_results.append(results)
        
        return all_results