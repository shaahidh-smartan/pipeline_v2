import cv2
import numpy as np
import os
import sys
import time

# Import face processing modules (now using utils-based versions)
from face_detection import SCRFD
from embedding_generator import ArcFace
from face_alignment import AlignCrop

# Add path to project root for utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.database_manager import DatabaseManager
from utils.onnx_loader import BaseONNXModel

def main():
    """
    Process images to detect faces, generate embeddings, and store them in database.

    This function implements a complete face registration pipeline that:
    1. Connects to PostgreSQL database for storage
    2. Detects faces in images using SCRFD model (now with utils base)
    3. Aligns and crops detected faces for consistent processing
    4. Generates 512-dimensional face embeddings using ArcFace model (now with utils base)
    5. Stores normalized embeddings in database with person names
    6. Creates visualizations showing detected faces and keypoints

    Input:
        - Images from 'temp_image/' directory (jpg, jpeg, png, bmp formats)
        - Uses pre-trained models: scrfd_10g_bnkps.onnx and mbf.onnx

    Output:
        - Aligned face images saved to 'aligned_faces/' directory
        - Detection visualizations saved to 'outputs/' directory
        - Face embeddings stored in PostgreSQL database
        - Console output showing processing progress and statistics

    Returns:
        None - Function prints results and saves files to disk

    Requirements:
        - GPU support for faster processing
        - Database connection configured in DatabaseManager
        - Model files in 'models/' directory
        - Utils package with BaseONNXModel
    """
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Test database connection
    success, message = db_manager.test_connection()
    if not success:
        print(f"[ERROR] Database connection failed: {message}")
        return
    print(f"[INFO] Database connection successful: {message}")
    
    # Face detector - Use GPU 0 (now inherits from BaseONNXModel)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..'))

    # Absolute model paths
    scrfd_path = os.path.join(repo_root, 'models', 'scrfd_10g_bnkps.onnx')
    mbf_path   = os.path.join(repo_root, 'models', 'mbf.onnx')

    detector = SCRFD(model_file=scrfd_path)
    detector.prepare(0)

    embedder = ArcFace(model_file=mbf_path)
    embedder.prepare(0)
    aligner = AlignCrop()
    # Absolute I/O dirs (optional but prevents similar surprises)
    input_dir   = os.path.join(repo_root, 'temp_image')
    output_dir  = os.path.join(repo_root, 'outputs')
    aligned_dir = os.path.join(repo_root, 'aligned_faces')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    
    # Process all images in the input directory
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found. Please create it and add some images.")
        return
    
    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        img_path = os.path.join(input_dir, img_file)
        print(f"\nProcessing: {img_file}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
        
        # Step 1: Face Detection
        print("  1. Detecting faces...")
        bboxes, kpss = detector.detect(img, thresh=0.5, input_size=(640, 640))
        
        if len(bboxes) == 0:
            print("  No faces detected.")
            continue
        
        print(f"  Found {len(bboxes)} face(s)")
        
        # Process each detected face
        for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
            print(f"  Processing face {i+1}/{len(bboxes)}")
            
            # Step 2: Face Alignment and Cropping
            print("    2. Aligning and cropping face...")
            try:
                aligned_face, tranform_matrix = aligner.align_and_crop(img, kps, image_size=112)
                
                # Save aligned face
                aligned_filename = f"{os.path.splitext(img_file)[0]}_face_{i+1}_aligned.jpg"
                aligned_path = os.path.join(aligned_dir, aligned_filename)
                cv2.imwrite(aligned_path, aligned_face)
                print(f"    Saved aligned face: {aligned_path}")
                
            except Exception as e:
                print(f"    Error in face alignment: {e}")
                continue
            
            # Step 3: Generate Face Embedding
            print("    3. Generating face embedding...")
            try:
                embedding = embedder.get_embedding(aligned_face)
                print(f"    Generated embedding with shape: {embedding.shape}")
                
                # Normalize embedding
                normalized_embedding = embedding / np.linalg.norm(embedding)
                
                # Step 4: Store in Database using DatabaseManager
                person_name = os.path.splitext(img_file)[0]
                created_at = int(time.time())
                
                success = db_manager.insert_face_embedding(
                    person_name=person_name,
                    embedding_vector=normalized_embedding,
                    created_at=created_at
                )
                
                if success:
                    print(f"    Successfully stored embedding for {person_name}")
                else:
                    print(f"    Failed to store embedding for {person_name}")

            except Exception as e:
                print(f"    Error in embedding generation: {e}")
                continue
        
        # Visualize detection results
        print("  4. Creating visualization...")
        vis_img = img.copy()
        
        for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
            # Draw bounding box
            x1, y1, x2, y2, score = bbox.astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(vis_img, f'{score:.3f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw keypoints
            if kps is not None:
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(vis_img, tuple(kp), 2, (0, 0, 255), -1)
        
        # Save visualization
        vis_filename = f"{os.path.splitext(img_file)[0]}_detection.jpg"
        vis_path = os.path.join(output_dir, vis_filename)
        cv2.imwrite(vis_path, vis_img)
        print(f"  Saved visualization: {vis_path}")
        
        print(f"  Completed processing {img_file}")
    
    # Display database statistics
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    
    stats = db_manager.get_database_stats()
    print(f"Face embeddings stored: {stats.get('face_people', 0)} people")
    print(f"Person embeddings stored: {stats.get('person_people', 0)} people")
    print(f"Total person embeddings: {stats.get('person_embeddings', 0)}")
    
    print("\nProcessing complete!")
    
    # Optional: Display model information from utils base class
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    detector_info = detector.get_model_info()
    embedder_info = embedder.get_model_info()
    
    print("SCRFD Detector:")
    print(f"  Model: {detector_info['model_file']}")
    print(f"  Providers: {detector_info['providers']}")
    print(f"  Input shape: {detector_info['inputs'][0]['shape']}")
    print(f"  Output count: {len(detector_info['outputs'])}")
    
    print("\nArcFace Embedder:")
    print(f"  Model: {embedder_info['model_file']}")
    print(f"  Providers: {embedder_info['providers']}")
    print(f"  Input shape: {embedder_info['inputs'][0]['shape']}")
    print(f"  Output shape: {embedder_info['outputs'][0]['shape']}")


if __name__ == '__main__':
    print("Face Registration Module")
    print("=" * 40)
    print("Processing images and generating face embeddings...")
    print("Now using optimized utils-based ONNX model loading!")
    print()
    
    # Run the main face registration function
    main()