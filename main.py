#!/usr/bin/env python3
"""
Person Embedding Collection Module - Main Entry Point

This module handles person embedding collection:
1. Monitor RTSP camera streams
2. Detect faces and match against stored face embeddings
3. When recognized persons are detected, collect their body embeddings
4. Store body embeddings in PostgreSQL database for re-identification

Usage:
    python person_embedding_collection_module/main.py
    cd person_embedding_collection_module && python main.py
"""

import os
import sys

# Fix libproxy crash with RTSP URLs containing special characters
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# Add parent directory to path to access utils and core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BpB_embedding_collector import PersonEmbeddingCollector


def create_camera_config():
    """
    Create configuration for 4 RTSP cameras.

    Args:
        None

    Returns:
        list: List of 4 camera configuration dictionaries
    """
    return [
        {
            'name': 'Right Camera',
            'url': 'rtsp://admin:admin%40123@192.168.0.101:554/stream1',
            'width': 640,
            'height': 640
        },
        {
            'name': 'Center Camera',
            'url': 'rtsp://admin:admin%40123@192.168.0.110:554/stream1',
            'width': 640,
            'height': 640
        }
    ]


def main():
    """
    Main function to run the person embedding collection system.

    Args:
        None

    Returns:
        None
    """
    SIMILARITY_THRESHOLD = 0.69
    TARGET_EMBEDDINGS_PER_PERSON = 25
    
    print("🎬 Person Embedding Collection Module")
    print("=" * 50)
    
    # Change to parent directory so relative paths work correctly
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)
    
    # Create outputs directory if it doesn't exist (for screenshots, etc.)
    os.makedirs("outputs", exist_ok=True)
    
    camera_configs = create_camera_config()
    
    print(f"Face Embeddings Source: PostgreSQL Database (face_embeddings table)")
    print(f"Person Embeddings Output: PostgreSQL Database (vgg_embeddings table)")
    print(f"Target Embeddings per Person: {TARGET_EMBEDDINGS_PER_PERSON} FRAMES")
    print()
    print("Camera Configuration:")
    for i, config in enumerate(camera_configs, 1):
        print(f"  Camera {i}: {config['name']} - {config['url']}")
    
    try:
        collector = PersonEmbeddingCollector(
            stream_configs=camera_configs,
            similarity_threshold=SIMILARITY_THRESHOLD,
            target_embeddings_per_person=TARGET_EMBEDDINGS_PER_PERSON
        )
        
        collector.start_embedding_collection()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()