import cv2
import numpy as np
import sys
import os

# Add project root to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.onnx_loader import BaseONNXModel


class ArcFace(BaseONNXModel):
    """
    ArcFace face embedding generator for face recognition.

    Creates high-quality 512-dimensional face embeddings that can be used for face
    recognition and similarity comparison. Uses deep learning model trained with
    ArcFace loss function for discriminative face representations.

    Key Features:
        - Generates 512-dimensional face embeddings
        - Thread-safe inference for concurrent processing
        - GPU acceleration support
        - L2 normalized embeddings for cosine similarity
        - Standard 112x112 input size for aligned faces
    """

    def __init__(self, model_file=None, session=None):
        """
        Initialize ArcFace embedding generator with ONNX model.

        Input:
            model_file (str, optional): Path to ONNX model file (e.g., 'mbf.onnx')
            session (onnxruntime.InferenceSession, optional): Pre-initialized ONNX session

        Output:
            None - Initializes the embedding generator object

        Returns:
            ArcFace instance ready for embedding generation after calling prepare()
        """
        # Set default input size before calling parent constructor
        self.input_size = (112, 112)  # Standard ArcFace input size
        
        super().__init__(model_file, session)
    
    def _initialize_model_info(self):
        """Initialize ArcFace-specific model information."""
        super()._initialize_model_info()
        
        # Update output name for single output model
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape and update input size if specified
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.input_size = tuple(input_shape[2:4])
    
    def get_embedding(self, aligned_face):
        """
        Generate 512-dimensional face embedding from aligned face image.

        Converts aligned face image to a compact numerical representation that
        captures facial features for recognition. The embedding is L2 normalized
        for cosine similarity calculations.

        Input:
            aligned_face (numpy.ndarray): Aligned face image, shape (H, W, 3) in BGR format

        Output:
            numpy.ndarray: Normalized face embedding vector of shape (512,)

        Returns:
            512-dimensional float array representing the face, normalized for comparison
        """
        # Preprocess the image
        blob = self._preprocess_image(aligned_face)
        
        # Perform thread-safe inference using parent class method
        embedding = self.safe_inference({self.input_name: blob})[0]
        
        # Normalize the embedding (L2 normalization)
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _preprocess_image(self, aligned_face):
        """
        Preprocess face image for ArcFace model input.
        
        Args:
            aligned_face (numpy.ndarray): Input face image
            
        Returns:
            numpy.ndarray: Preprocessed image blob
        """
        # Ensure the image is the correct size
        if aligned_face.shape[:2] != self.input_size:
            aligned_face = cv2.resize(aligned_face, self.input_size)
        
        # Convert BGR to RGB if needed
        if len(aligned_face.shape) == 3 and aligned_face.shape[2] == 3:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [-1, 1] range
        aligned_face = aligned_face.astype(np.float32)
        aligned_face = (aligned_face - 127.5) / 127.5
        
        # Add batch dimension and transpose to NCHW format
        blob = np.transpose(aligned_face, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        return blob
    
    def get_embeddings(self, aligned_faces):
        """
        Generate embeddings for multiple aligned face images in batch.

        Processes a list of aligned face images and returns corresponding embeddings.
        Each face is processed individually through the neural network to generate
        its unique 512-dimensional representation.

        Input:
            aligned_faces (list): List of numpy arrays, each shape (H, W, 3) in BGR format

        Output:
            list: List of normalized embedding vectors, each shape (512,)

        Returns:
            List of 512-dimensional float arrays, one embedding per input face
        """
        embeddings = []
        for face in aligned_faces:
            embedding = self.get_embedding(face)
            embeddings.append(embedding)
        return embeddings