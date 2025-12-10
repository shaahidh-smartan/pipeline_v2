#!/usr/bin/env python3
"""
Build gallery bank from video files instead of static images.
Extracts frames from videos and generates embeddings.
"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from torchreid.scripts.main import build_config
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.utils.constants import bn_correspondants


class SimpleArgs:
    """Minimal args object for build_config compatibility."""
    def __init__(self):
        self.root = ''
        self.save_dir = 'log'
        self.job_id = 'gallery_build'
        self.inference_enabled = False
        self.sources = None
        self.targets = None
        self.transforms = None
        self.opts = []


def extract_test_embeddings(model_output: tuple, cfg: object) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and concatenate test embeddings from model output."""
    embeddings_dict, visibility_dict, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output

    embeddings_list = []
    visibility_scores_list = []

    for test_emb in cfg.model.bpbreid.test_embeddings:
        embds = embeddings_dict[test_emb]
        embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))

        vis_key = test_emb
        if test_emb in bn_correspondants:
            vis_key = bn_correspondants[test_emb]

        vis_scores = visibility_dict[vis_key]
        visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))

    embeddings = torch.cat(embeddings_list, dim=1)
    visibility_scores = torch.cat(visibility_scores_list, dim=1)

    return embeddings, visibility_scores


def extract_frames_from_video(video_path: Path,
                               frame_interval: int = 30,
                               max_frames: int = 50,
                               yolo_model=None) -> List[np.ndarray]:
    """
    Extract person crops from video using YOLO detection.

    Args:
        video_path: Path to video file
        frame_interval: Extract one frame every N frames
        max_frames: Maximum number of person crops to extract
        yolo_model: YOLO model for person detection

    Returns:
        List of person crop frames as RGB numpy arrays
    """
    from ultralytics import YOLO

    if yolo_model is None:
        yolo_model = YOLO('yolo11n.pt')

    cap = cv2.VideoCapture(str(video_path))
    person_crops = []
    frame_idx = 0

    while len(person_crops) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Run YOLO detection
            results = yolo_model(frame, classes=[0], verbose=False)  # class 0 = person

            # Find largest person (same as inference)
            largest_area = 0
            largest_box = None

            for det in results[0].boxes:
                conf = float(det.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, det.xyxy[0])

                # Filter small detections
                if (y2 - y1) < 50 or (x2 - x1) < 20:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (x1, y1, x2, y2)

            if largest_box:
                x1, y1, x2, y2 = largest_box

                # Crop person from frame
                crop = frame[y1:y2, x1:x2]

                # Convert BGR to RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                person_crops.append(crop_rgb)

        frame_idx += 1

    cap.release()
    return person_crops


def extract_features_for_person(frames: List[np.ndarray],
                                extractor: FeatureExtractor,
                                cfg: object,
                                device: str,
                                batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract BPBreID features from video frames.

    Args:
        frames: List of RGB numpy arrays
        extractor: FeatureExtractor instance
        cfg: Config object
        device: Device string
        batch_size: Batch size for processing

    Returns:
        (embeddings, visibility_scores) tuple
    """
    if not frames:
        return None, None

    all_embeddings = []
    all_visibility = []

    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]

        # Extract features
        model_output = extractor(batch_frames)
        embeddings, visibility_scores = extract_test_embeddings(model_output, cfg)

        all_embeddings.append(embeddings.to(device))
        all_visibility.append(visibility_scores.to(device))

    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    visibility = torch.cat(all_visibility, dim=0)

    return embeddings, visibility


def build_gallery_from_videos(config_path: str,
                              videos_dir: str,
                              output_dir: str,
                              weights_path: str = None,
                              device: str = 'cuda',
                              frame_interval: int = 30,
                              max_frames_per_video: int = 50,
                              batch_size: int = 32,
                              aggregation: str = 'mean'):
    """
    Build gallery bank from video files.

    Args:
        config_path: Path to BPBreID config
        videos_dir: Directory containing person video subdirectories
        output_dir: Directory to save gallery files
        weights_path: Optional weights path
        device: Device to use
        frame_interval: Extract one frame every N frames
        max_frames_per_video: Maximum frames to extract per video
        batch_size: Batch size for processing
        aggregation: 'mean' or 'concat'
    """
    print("="*80)
    print("Building Gallery Bank from Videos")
    print("="*80)

    # Build config and extractor
    print("\n[1/4] Loading config and model...")
    dummy_args = SimpleArgs()
    cfg = build_config(args=dummy_args, config_file=config_path)
    cfg.use_gpu = (device.startswith('cuda') and torch.cuda.is_available())

    model_path = weights_path or cfg.model.load_weights
    extractor = FeatureExtractor(
        cfg,
        model_path=model_path,
        device=device if torch.cuda.is_available() else 'cpu',
        num_classes=1,
        verbose=True
    )
    device = extractor.device

    # Find person video directories
    print("\n[2/4] Scanning video directories...")
    videos_path = Path(videos_dir)
    person_dirs = sorted([d for d in videos_path.iterdir() if d.is_dir()])

    if not person_dirs:
        print(f"ERROR: No person directories found in {videos_dir}")
        return

    print(f"Found {len(person_dirs)} person directories:")
    for person_dir in person_dirs:
        videos = list(person_dir.glob('*.mp4')) + list(person_dir.glob('*.avi'))
        print(f"  - {person_dir.name}: {len(videos)} videos")

    # Extract features for each person
    print("\n[3/4] Extracting features from videos...")
    all_embeddings = []
    all_visibility = []
    all_pids = []

    for pid, person_dir in enumerate(person_dirs, start=1):
        print(f"\nProcessing Person {pid} ({person_dir.name})...")

        # Find all videos for this person
        video_files = sorted(list(person_dir.glob('*.mp4')) + list(person_dir.glob('*.avi')))

        if not video_files:
            print(f"   Warning: No videos found for {person_dir.name}")
            continue

        person_embeddings = []
        person_visibility = []

        for video_file in tqdm(video_files, desc=f"  Extracting from videos"):
            # Extract frames
            frames = extract_frames_from_video(
                video_file,
                frame_interval=frame_interval,
                max_frames=max_frames_per_video
            )

            if not frames:
                print(f"   Warning: No frames extracted from {video_file.name}")
                continue

            # Extract features
            embeddings, visibility = extract_features_for_person(
                frames, extractor, cfg, device, batch_size
            )

            if embeddings is not None:
                person_embeddings.append(embeddings)
                person_visibility.append(visibility)

        if not person_embeddings:
            print(f"   Warning: No features extracted for {person_dir.name}")
            continue

        # Concatenate all videos for this person
        person_embeddings = torch.cat(person_embeddings, dim=0)
        person_visibility = torch.cat(person_visibility, dim=0)

        print(f"   Extracted {person_embeddings.shape[0]} frames total")

        # Aggregate if needed
        if aggregation == 'mean':
            person_embeddings = person_embeddings.mean(dim=0, keepdim=True)
            if person_visibility.dtype == torch.bool:
                person_visibility = person_visibility.float()
            person_visibility = person_visibility.mean(dim=0, keepdim=True)
            print(f"   Aggregated to single representation (mean)")

        all_embeddings.append(person_embeddings)
        all_visibility.append(person_visibility)

        # Create PIDs
        num_samples = person_embeddings.shape[0]
        all_pids.extend([pid] * num_samples)

    # Concatenate all persons
    gallery_embeddings = torch.cat(all_embeddings, dim=0)
    gallery_visibility = torch.cat(all_visibility, dim=0)
    gallery_pids = torch.tensor(all_pids, dtype=torch.long, device=device)

    # Save gallery
    print(f"\n[4/4] Saving gallery to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(gallery_embeddings, output_path / 'gallery_embeddings.pt')
    torch.save(gallery_visibility, output_path / 'gallery_visibility.pt')
    torch.save(gallery_pids, output_path / 'gallery_pids.pt')

    print("\n" + "="*80)
    print("Gallery Bank Summary")
    print("="*80)
    print(f"Total samples: {gallery_embeddings.shape[0]}")
    print(f"Embedding shape: {gallery_embeddings.shape}")
    print(f"Visibility shape: {gallery_visibility.shape}")
    print(f"Person IDs: {torch.unique(gallery_pids).tolist()}")
    print(f"\nGallery saved to: {output_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Build BPBreID gallery from video files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to BPBreID YAML config file')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to weights file (if not in config)')
    parser.add_argument('--videos-dir', type=str, required=True,
                       help='Directory containing person video subdirectories')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save gallery files')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Extract one frame every N frames')
    parser.add_argument('--max-frames-per-video', type=int, default=50,
                       help='Maximum frames to extract per video')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'concat'],
                       help='How to aggregate frames: mean or concat')

    args = parser.parse_args()

    build_gallery_from_videos(
        config_path=args.config,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        weights_path=args.weights,
        device=args.device,
        frame_interval=args.frame_interval,
        max_frames_per_video=args.max_frames_per_video,
        batch_size=args.batch_size,
        aggregation=args.aggregation
    )


if __name__ == '__main__':
    main()
