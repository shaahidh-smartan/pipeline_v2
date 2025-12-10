#!/bin/bash
# Quick script to generate gallery embeddings from videos

set -e  # Exit on error

echo "========================================================================"
echo "BPBreID Gallery Generation Script"
echo "========================================================================"

# Default values
VIDEOS_DIR="syb_embedding"
OUTPUT_DIR="gallery_bank_syb_1"
CONFIG="configs/test_reid.yaml"
FRAME_INTERVAL=10
MAX_FRAMES=15
AGGREGATION="concat"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --videos-dir)
            VIDEOS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --aggregation)
            AGGREGATION="$2"
            shift 2
            ;;
        --clean)
            echo "Clean mode: Will delete existing gallery and logs"
            CLEAN_MODE=1
            shift
            ;;
        -h|--help)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --videos-dir DIR       Directory with person video folders (default: syb_embedding)"
            echo "  --output-dir DIR       Output directory for gallery (default: gallery_bank_vijay)"
            echo "  --frame-interval N     Extract one frame every N frames (default: 10)"
            echo "  --max-frames N         Maximum frames per video (default: 15)"
            echo "  --aggregation TYPE     concat or mean (default: concat)"
            echo "  --clean                Delete existing gallery before generation"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Use defaults"
            echo "  $0 --clean                      # Clean and regenerate"
            echo "  $0 --max-frames 30              # More samples per person"
            echo "  $0 --frame-interval 5           # Sample more frequently"
            echo "  $0 --aggregation mean           # Average embeddings"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if videos directory exists
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "ERROR: Videos directory not found: $VIDEOS_DIR"
    echo "Please create it and add person video folders"
    exit 1
fi

# Check if videos directory has subdirectories
if [ -z "$(ls -A $VIDEOS_DIR)" ]; then
    echo "ERROR: Videos directory is empty: $VIDEOS_DIR"
    echo "Please add person folders with videos"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Videos directory:  $VIDEOS_DIR"
echo "  Output directory:  $OUTPUT_DIR"
echo "  Config file:       $CONFIG"
echo "  Frame interval:    $FRAME_INTERVAL"
echo "  Max frames/video:  $MAX_FRAMES"
echo "  Aggregation:       $AGGREGATION"
echo ""

# List person folders
echo "Person folders found:"
for dir in "$VIDEOS_DIR"/*; do
    if [ -d "$dir" ]; then
        person_name=$(basename "$dir")
        video_count=$(find "$dir" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" \) | wc -l)
        echo "  - $person_name: $video_count videos"
    fi
done
echo ""

# Clean if requested
if [ "$CLEAN_MODE" = "1" ]; then
    echo "Cleaning old gallery and logs..."
    if [ -d "$OUTPUT_DIR" ]; then
        rm -rf "$OUTPUT_DIR"
        echo "  Deleted: $OUTPUT_DIR"
    fi
    if [ -d "log/gallery_build" ]; then
        rm -rf log/gallery_build
        echo "  Deleted: log/gallery_build"
    fi
    echo ""
fi

# Check if gallery already exists
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/gallery_embeddings.pt" ]; then
    echo "WARNING: Gallery already exists at $OUTPUT_DIR"
    echo "Use --clean to delete and regenerate"
    echo ""
    read -p "Continue anyway? This will overwrite. (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ ! -f "bpbreid_venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found: bpbreid_venv"
    echo "Please create it first: python3 -m venv bpbreid_venv"
    exit 1
fi
source bpbreid_venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run gallery generation
echo "========================================================================"
echo "Generating gallery embeddings..."
echo "========================================================================"
echo ""

python3 build_gallery_from_videos.py \
    --config "$CONFIG" \
    --videos-dir "$VIDEOS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --aggregation "$AGGREGATION" \
    --frame-interval "$FRAME_INTERVAL" \
    --max-frames-per-video "$MAX_FRAMES"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Gallery generation completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Gallery saved to: $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    if [ -f "$OUTPUT_DIR/gallery_embeddings.pt" ]; then
        emb_size=$(du -h "$OUTPUT_DIR/gallery_embeddings.pt" | cut -f1)
        echo "  ✓ gallery_embeddings.pt  ($emb_size)"
    fi
    if [ -f "$OUTPUT_DIR/gallery_visibility.pt" ]; then
        vis_size=$(du -h "$OUTPUT_DIR/gallery_visibility.pt" | cut -f1)
        echo "  ✓ gallery_visibility.pt  ($vis_size)"
    fi
    if [ -f "$OUTPUT_DIR/gallery_pids.pt" ]; then
        pid_size=$(du -h "$OUTPUT_DIR/gallery_pids.pt" | cut -f1)
        echo "  ✓ gallery_pids.pt        ($pid_size)"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Test gallery:       python3 test_embedding_inference.py"
    echo "  2. Video inference:    python3 video_reid_clean.py --gallery-dir $OUTPUT_DIR --video <video.mp4>"
    echo "  3. RTSP inference:     python3 rtsp_reid_inference.py --gallery-dir $OUTPUT_DIR --rtsp-url <url>"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "✗ Gallery generation failed"
    echo "========================================================================"
    echo ""
    echo "Please check the error messages above"
    exit 1
fi
