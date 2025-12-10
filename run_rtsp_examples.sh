#!/bin/bash
# RTSP ReID Inference - Example Commands
# This script shows various configurations for running RTSP inference

# Activate virtual environment
source bpbreid_venv/bin/activate

# ==============================================================================
# EXAMPLE 1: Basic RTSP inference with default settings
# ==============================================================================
echo "Example 1: Basic inference with defaults"
echo "=========================================="
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream"

# ==============================================================================
# EXAMPLE 2: Adjust matching threshold (more/less strict)
# ==============================================================================
echo ""
echo "Example 2: Strict matching (lower threshold)"
echo "============================================="
# Lower threshold = stricter matching (only very close matches accepted)
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --matching-threshold 4.0

# Higher threshold = more lenient (accept more matches)
# --matching-threshold 8.0

# ==============================================================================
# EXAMPLE 3: Adjust voting parameters for faster/slower caching
# ==============================================================================
echo ""
echo "Example 3: Fast caching (fewer votes needed)"
echo "============================================="
# Faster caching: smaller window, fewer votes required
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --voting-window 20 \
  --min-votes 10 \
  --voting-threshold 4.5

# Slower but more accurate caching: larger window, more votes
# --voting-window 50 --min-votes 30 --voting-threshold 4.0

# ==============================================================================
# EXAMPLE 4: Top-K ranking - show multiple possible matches
# ==============================================================================
echo ""
echo "Example 4: Show top-3 matches for each person"
echo "=============================================="
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --top-k 3 \
  --show-all-ranks

# This will display all 3 top matches below each detected person

# ==============================================================================
# EXAMPLE 5: Adjust YOLO detection parameters
# ==============================================================================
echo ""
echo "Example 5: More sensitive person detection"
echo "==========================================="
# Lower confidence = detect more people (may have false positives)
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --yolo-conf 0.3 \
  --yolo-min-height 30 \
  --yolo-min-width 15

# Higher confidence = detect only clear people (may miss some)
# --yolo-conf 0.7 --yolo-min-height 100 --yolo-min-width 40

# ==============================================================================
# EXAMPLE 6: Performance optimization - frame skipping
# ==============================================================================
echo ""
echo "Example 6: Skip frames for faster processing"
echo "============================================="
# Process every 2nd frame (skip 1 frame)
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --frame-skip 1

# Process every 3rd frame (skip 2 frames) for even faster processing
# --frame-skip 2

# ==============================================================================
# EXAMPLE 7: Save output video without display
# ==============================================================================
echo ""
echo "Example 7: Record output without display (headless mode)"
echo "========================================================="
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --output output_rtsp.mp4 \
  --no-display

# ==============================================================================
# EXAMPLE 8: Complete custom configuration
# ==============================================================================
echo ""
echo "Example 8: Full custom configuration"
echo "====================================="
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream" \
  --output monitored_stream.mp4 \
  --voting-threshold 4.0 \
  --voting-window 40 \
  --min-votes 20 \
  --matching-threshold 5.5 \
  --yolo-conf 0.6 \
  --yolo-min-height 60 \
  --yolo-min-width 25 \
  --top-k 2 \
  --show-all-ranks \
  --frame-skip 0 \
  --reconnect-delay 3.0 \
  --device cuda

# ==============================================================================
# PARAMETER REFERENCE
# ==============================================================================
cat <<'EOF'

=============================================================================
PARAMETER REFERENCE
=============================================================================

VOTING & CACHING PARAMETERS:
----------------------------
--voting-threshold <float>   Distance threshold for counting votes (default: 4.5)
                              Lower = stricter voting, higher = more lenient

--voting-window <int>         Number of frames to collect votes (default: 30)
                              Smaller = faster caching, larger = more stable

--min-votes <int>             Minimum votes to cache an ID (default: 15)
                              Lower = faster caching, higher = more confident

--matching-threshold <float>  Display matching threshold (default: 6.0)
                              Distance > threshold shown as "Unknown"
                              Lower = stricter recognition, higher = more lenient

YOLO DETECTION PARAMETERS:
--------------------------
--yolo-conf <float>           YOLO confidence threshold (default: 0.5)
                              Lower = detect more (more false positives)
                              Higher = detect only clear persons

--yolo-min-height <int>       Minimum person height in pixels (default: 50)
                              Filter out small/distant detections

--yolo-min-width <int>        Minimum person width in pixels (default: 20)
                              Filter out thin/partial detections

RANKING & DISPLAY:
------------------
--top-k <int>                 Number of top matches to compute (default: 1)
                              Set to 3-5 to see alternative matches

--show-all-ranks              Display all top-k rankings on video
                              Shows ranked list below each person

PERFORMANCE & CONNECTION:
-------------------------
--frame-skip <int>            Skip N frames between processing (default: 0)
                              0 = process all frames
                              1 = process every 2nd frame (faster)
                              2 = process every 3rd frame (even faster)

--reconnect-delay <float>     Seconds to wait before reconnecting (default: 5.0)
                              Time to wait after connection loss

--device <cuda|cpu>           Device to use (default: cuda)

--output <path>               Save output video to file (optional)

--no-display                  Disable live display window

=============================================================================
RECOMMENDED CONFIGURATIONS
=============================================================================

HIGH ACCURACY (Slower, more stable):
  --voting-threshold 4.0 --voting-window 50 --min-votes 30
  --matching-threshold 5.0 --yolo-conf 0.6

BALANCED (Default):
  --voting-threshold 4.5 --voting-window 30 --min-votes 15
  --matching-threshold 6.0 --yolo-conf 0.5

FAST RESPONSE (Faster, less stable):
  --voting-threshold 5.0 --voting-window 20 --min-votes 10
  --matching-threshold 7.0 --yolo-conf 0.4 --frame-skip 1

DEBUGGING MODE:
  --top-k 5 --show-all-ranks
  (Shows top 5 matches for analysis)

=============================================================================
EOF
