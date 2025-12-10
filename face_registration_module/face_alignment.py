import cv2
import numpy as np

class AlignCrop:
    """
    Face alignment and cropping utility for face recognition preprocessing.

    Aligns detected faces to a standardized orientation and crops them to a fixed size.
    Uses facial landmarks (eye, nose, mouth positions) to calculate alignment
    transformation that normalizes pose, scale, and rotation variations.

    Key Features:
        - Aligns faces using 5-point facial landmarks
        - Supports 112x112 and 128x128 output sizes
        - Uses ArcFace reference landmark positions
        - Applies affine transformation for geometric normalization
        - Returns both aligned image and transformation matrix
    """

    def __init__(self):
        """
        Initialize face alignment and cropping processor.

        Input:
            None - No parameters required for initialization

        Output:
            None - Creates ready-to-use AlignCrop instance

        Returns:
            AlignCrop object ready for face alignment operations
        """
        pass
    
    def align_and_crop(self, img, landmarks, image_size=112):
        """
        Align and crop face to standardized pose using facial landmarks.

        Takes a full image and 5 facial landmarks to compute alignment transformation.
        Warps the face to match reference landmark positions used in ArcFace training,
        ensuring consistent face orientation for embedding generation.

        Input:
            img (numpy.ndarray): Full input image in BGR format, shape (H, W, 3)
            landmarks (numpy.ndarray or list): 5 facial keypoints as (x, y) coordinates
                                             [left_eye, right_eye, nose, left_mouth, right_mouth]
            image_size (int): Output image size in pixels, must be 112 or 128

        Output:
            tuple: (aligned_face, transformation_matrix)

        Returns:
            aligned_face: Cropped and aligned face image, shape (image_size, image_size, 3)
            transformation_matrix: 2x3 affine transformation matrix used for alignment

        Raises:
            ValueError: If transformation matrix cannot be estimated
            AssertionError: If landmarks count != 5 or invalid image_size
        """
        # Define the reference keypoints used in ArcFace model, based on a typical facial landmark set.
        arcface_ref_kps = np.array(
            [
                [38.2946, 51.6963],  # Left eye
                [73.5318, 51.5014],  # Right eye
                [56.0252, 71.7366],  # Nose
                [41.5493, 92.3655],  # Left mouth corner
                [70.7299, 92.2041],  # Right mouth corner
            ],
            dtype=np.float32,
        )
        
        # Ensure the input landmarks have exactly 5 points (as expected for face alignment)
        assert len(landmarks) == 5, f"Expected 5 landmarks, got {len(landmarks)}"
        
        # Validate that image_size is divisible by either 112 or 128 (common image sizes for face recognition models)
        assert image_size % 112 == 0 or image_size % 128 == 0, f"Image size {image_size} must be divisible by 112 or 128"
        
        # Adjust the scaling factor (ratio) based on the desired image size (112 or 128)
        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0  # No horizontal shift for 112 scaling
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio  # Horizontal shift for 128 scaling
        
        # Apply the scaling and shifting to the reference keypoints
        dst = arcface_ref_kps * ratio
        dst[:, 0] += diff_x  # Apply the horizontal shift
        
        # Convert landmarks to numpy array if it's a list
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        # Ensure landmarks are in the correct shape (N, 2)
        if landmarks.shape != (5, 2):
            landmarks = landmarks.reshape(5, 2)
        
        # Estimate the similarity transformation matrix to align the landmarks with the reference keypoints
        M, inliers = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32), 
            dst, 
            ransacReprojThreshold=1000
        )
        
        # Check if the transformation was successful
        if M is None:
            raise ValueError("Failed to estimate affine transformation matrix")
        
        # Verify that all points were used as inliers (optional, can be commented out for robustness)
        # assert np.all(inliers == True), "Not all landmarks were used as inliers in the transformation"
        
        # Apply the affine transformation to the input image to align the face
        aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        
        return aligned_img, M