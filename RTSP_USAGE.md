# RTSP ReID Inference - Usage Guide

Complete guide for configuring RTSP person re-identification inference with command-line parameters.

## Quick Start

```bash
source bpbreid_venv/bin/activate

python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://your_camera_ip:554/stream"
```

## Command-Line Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Path to BPBreID YAML config file |
| `--gallery-dir` | Directory containing gallery embeddings (`.pt` files) |
| `--rtsp-url` | RTSP stream URL from your camera |

### Voting & Caching Parameters

Control how person IDs are cached over time using voting mechanism.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--voting-threshold` | 4.5 | Distance threshold for counting a vote. Lower = stricter. |
| `--voting-window` | 30 | Number of frames to collect votes. Smaller = faster caching. |
| `--min-votes` | 15 | Minimum votes needed to cache an ID. Lower = faster, higher = more confident. |
| `--matching-threshold` | 6.0 | Display threshold: distance > this shows "Unknown". Lower = stricter recognition. |

**Example - Strict caching:**
```bash
--voting-threshold 4.0 --voting-window 50 --min-votes 30 --matching-threshold 5.0
```

**Example - Fast caching:**
```bash
--voting-threshold 5.0 --voting-window 20 --min-votes 10 --matching-threshold 7.0
```

### YOLO Detection Parameters

Control person detection sensitivity and filtering.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yolo-conf` | 0.5 | YOLO confidence threshold. Lower = detect more (may have false positives). |
| `--yolo-min-height` | 50 | Minimum person height in pixels. Filters small/distant detections. |
| `--yolo-min-width` | 20 | Minimum person width in pixels. Filters thin/partial detections. |

**Example - Sensitive detection (detect more people):**
```bash
--yolo-conf 0.3 --yolo-min-height 30 --yolo-min-width 15
```

**Example - Conservative detection (only clear persons):**
```bash
--yolo-conf 0.7 --yolo-min-height 100 --yolo-min-width 40
```

### Ranking & Display Parameters

Show multiple possible matches for analysis or debugging.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 1 | Number of top matches to compute. Set to 3-5 to see alternatives. |
| `--show-all-ranks` | False | Display all top-k rankings on video below each person. |

**Example - Show top 3 matches:**
```bash
--top-k 3 --show-all-ranks
```

### Performance & Connection Parameters

Optimize processing speed and handle connection issues.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--frame-skip` | 0 | Skip N frames between processing. 0 = all, 1 = every 2nd, 2 = every 3rd. |
| `--reconnect-delay` | 5.0 | Seconds to wait before reconnecting after connection loss. |
| `--device` | cuda | Device to use: `cuda` or `cpu`. |
| `--output` | None | Optional path to save output video. |
| `--no-display` | False | Disable live display window (headless mode). |

**Example - Faster processing:**
```bash
--frame-skip 1 --device cuda
```

**Example - Save video without display:**
```bash
--output output.mp4 --no-display
```

## Complete Examples

### Example 1: Default Settings
```bash
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://192.168.1.100:554/stream"
```

### Example 2: High Accuracy (Slower, More Stable)
```bash
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://192.168.1.100:554/stream" \
  --voting-threshold 4.0 \
  --voting-window 50 \
  --min-votes 30 \
  --matching-threshold 5.0 \
  --yolo-conf 0.6
```

### Example 3: Fast Response (Faster, Less Stable)
```bash
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://192.168.1.100:554/stream" \
  --voting-threshold 5.0 \
  --voting-window 20 \
  --min-votes 10 \
  --matching-threshold 7.0 \
  --yolo-conf 0.4 \
  --frame-skip 1
```

### Example 4: Debugging Mode (Show Top 5 Matches)
```bash
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://192.168.1.100:554/stream" \
  --top-k 5 \
  --show-all-ranks
```

### Example 5: Production Mode (Record Without Display)
```bash
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://192.168.1.100:554/stream" \
  --output monitored_stream.mp4 \
  --no-display \
  --frame-skip 1
```

### Example 6: Custom Configuration
```bash
python3 rtsp_reid_inference.py \
  --config configs/test_reid.yaml \
  --gallery-dir gallery_bank_vijay \
  --rtsp-url "rtsp://192.168.1.100:554/stream" \
  --output output.mp4 \
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
  --reconnect-delay 3.0
```

## Understanding Distance Thresholds

The system uses Euclidean distance to compare person embeddings:

- **Distance 0-3**: Very strong match (same person, similar pose/lighting)
- **Distance 3-6**: Good match (same person, different conditions)
- **Distance 6-10**: Weak match (uncertain, may be different person)
- **Distance >10**: No match (definitely different person)

### Threshold Tuning Guide

**`--voting-threshold`** (default 4.5):
- Controls which matches count as "votes" for caching
- Lower (3.0-4.0) = Only very confident matches vote → slower but more accurate caching
- Higher (5.0-6.0) = More matches vote → faster but may cache wrong IDs

**`--matching-threshold`** (default 6.0):
- Controls display: distance > threshold shows "Unknown"
- Lower (4.0-5.0) = Stricter recognition, more "Unknown" labels → fewer false positives
- Higher (7.0-8.0) = More lenient, fewer "Unknown" labels → more false positives

**Recommended combinations:**
- **Conservative**: `--voting-threshold 4.0 --matching-threshold 5.0`
- **Balanced** (default): `--voting-threshold 4.5 --matching-threshold 6.0`
- **Lenient**: `--voting-threshold 5.0 --matching-threshold 7.0`

## Display Legend

When running with display enabled, you'll see:

- **Blue box + asterisk (*)**: Cached ID (high confidence from voting)
- **Green box**: Recognized person (distance < matching_threshold)
- **Red box + "UNK"**: Unknown person (distance > matching_threshold)

Format: `T<track_id>|ID<person_id> d=<distance>`

Example: `T2|ID1* d=3.45` = Track 2, cached as Person ID 1, distance 3.45

## Tips & Best Practices

1. **Start with defaults** and adjust based on performance
2. **Monitor distances** in console output to tune thresholds
3. **Use `--show-all-ranks`** when debugging to see alternative matches
4. **Increase `--frame-skip`** if processing is too slow
5. **Lower `--yolo-conf`** if missing person detections
6. **Increase `--voting-window` and `--min-votes`** for more stable IDs in crowded scenes
7. **Use `--no-display` and `--output`** for headless servers

## See Also

- [run_rtsp_examples.sh](run_rtsp_examples.sh) - Executable script with all examples
- [build_gallery_from_videos.py](build_gallery_from_videos.py) - Build gallery from videos
- [rtsp_reid_inference.py](rtsp_reid_inference.py) - Main inference script
