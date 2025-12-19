#!/usr/bin/env python3
"""
Single Camera Person Embedding Collection Module - Main Entry Point

This module handles person embedding collection using two cameras:
1. Primary camera: Face recognition only
2. Secondary camera: Body embedding collection only

When a face is recognized in the primary camera, the system triggers
collection from the secondary camera. Embeddings are always overwritten.

Usage:
    python person_embedding_collection_module/main_single_camera.py
    cd person_embedding_collection_module && python main_single_camera.py
"""

import os
import sys

# Fix libproxy crash with RTSP URLs containing special characters
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# Add parent directory to path to access utils and core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from single_camera_collector import SingleCameraCollector


def create_camera_config():
    """
    Create configuration for single camera (both face recognition and body embedding).

    Returns:
        dict: Camera configuration
    """
    # Single camera - for both face recognition and body embedding collection
    camera_config = {
        'name': 'Primary Camera',
        'url': 'rtsp://admin:admin%40123@192.168.0.110:554/stream1',
        'width': 640,
        'height': 640
    }

    return camera_config


def main():
    """
    Main function to run the single camera person embedding collection system.
    """
    SIMILARITY_THRESHOLD = 0.70
    TARGET_EMBEDDINGS_PER_PERSON = 25

    print("🎬 Single Camera Person Embedding Collection Module")
    print("=" * 60)

    # Change to parent directory so relative paths work correctly
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)

    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    camera_config = create_camera_config()

    print(f"Face Embeddings Source: PostgreSQL Database (face_embeddings table)")
    print(f"Person Embeddings Output: .PT files + embedding_status table")
    print(f"Target Embeddings per Person: {TARGET_EMBEDDINGS_PER_PERSON} FRAMES")
    print()
    print("Camera Configuration:")
    print(f"  Camera: {camera_config['name']} - {camera_config['url']}")
    print()
    print("Status-Based Collection Behavior:")
    print("  - Status 'READY': Collection BLOCKED - person can proceed to gym")
    print("  - Status 'CONSUMED': Collection ALLOWED - person has been processed")
    print("  - Status 'PENDING' or None: Collection ALLOWED - incomplete or new")
    print()
    print("Workflow:")
    print("  1. Camera recognizes face")
    print("  2. Check embedding_status in database")
    print("  3. If status is 'READY' -> Person can go to gym (no collection)")
    print("  4. If status is NOT 'READY' -> Collect embeddings from same camera")
    print("  5. After collection -> Set status to 'READY'")
    print("  6. Gym system sets status to 'CONSUMED' after person completes workout")

    try:
        collector = SingleCameraCollector(
            camera_config=camera_config,
            similarity_threshold=SIMILARITY_THRESHOLD,
            target_embeddings_per_person=TARGET_EMBEDDINGS_PER_PERSON
        )

        collector.start_collection()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
