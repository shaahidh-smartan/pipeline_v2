# BPBreID Distance Threshold Calculation - Complete Explanation

## Overview
The threshold in your ReID system determines whether a detected person matches someone in the gallery. The distance calculation involves multiple steps that combine body part features with visibility scores.

---

## Complete Pipeline

### 1. **Feature Extraction**
When a person is detected, BPBreID extracts features for different body parts:

```
Input: Person crop (RGB image)
      ↓
BPBreID Model
      ↓
Output:
  - Embeddings: [N, P+2, D]
    - N = number of people
    - P+2 = 6 body parts (global, foreground, 4 body parts)
    - D = 512 (embedding dimension)
  - Visibility Scores: [N, P+2]
    - Float values indicating how visible each part is
```

**In your config:**
- `test_embeddings: ['bn_foreg', 'parts']`
- This extracts 6 parts total: 1 global + 1 foreground + 4 body parts

### 2. **Body Part Distance Computation**
For each body part, compute Euclidean distance between query and gallery:

```python
# From torchreid/metrics/distance.py:222-236
def _compute_body_parts_dist_matrices(qf, gf, metric='euclidean'):
    # qf = query features [Nq, P+2, D]
    # gf = gallery features [Ng, P+2, D]

    # For each part p:
    # distance[p] = sqrt(||query[p] - gallery[p]||^2)

    # Mathematical formula:
    # ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2

    distances = qf_square_sum - 2 * dot_product + gf_square_sum
    distances = sqrt(ReLU(distances))  # ReLU to avoid negatives from numerical errors

    return distances  # Shape: [P+2, Nq, Ng]
```

**Example:**
- Query person: [global_feat, foreground_feat, part1, part2, part3, part4]
- Gallery person: Same structure
- Result: 6 distances (one per body part)

### 3. **Visibility-Weighted Distance Combination**

Since your visibility scores are **floats** (not boolean), the system uses:

```python
# From torchreid/metrics/distance.py:176-208
def _compute_distance_matrix_using_bp_features_and_visibility_scores(...):
    # Step 1: Compute per-part distances
    batch_body_part_pairwise_dist = _compute_body_parts_dist_matrices(qf, batch_gf, 'euclidean')
    # Shape: [6, Nq, Ng]

    # Step 2: Compute visibility weights
    # Multiply query visibility × gallery visibility for each part
    vis_scores = qf_parts_visibility.unsqueeze(2) * batch_gf_parts_visibility.unsqueeze(1)
    # Shape: [6, Nq, Ng]

    # Step 3: Combine distances using visibility-weighted mean
    if dist_combine_strat == 'mean':
        batch_pairwise_dist = visibility_masked_mean(
            batch_body_part_pairwise_dist,
            vis_scores
        )
        # Weighted mean = sum(distance × visibility) / sum(visibility)
```

**Mathematical Formula:**
```
final_distance = Σ(distance[p] × vis_query[p] × vis_gallery[p])
                 / Σ(vis_query[p] × vis_gallery[p])

where p ∈ {global, foreground, part1, part2, part3, part4}
```

**Example Calculation:**
```
Part distances:     [2.5, 3.0, 8.0, 7.5, 6.0, 9.0]
Query visibility:   [1.0, 0.9, 0.8, 0.3, 0.7, 0.2]
Gallery visibility: [1.0, 0.8, 0.9, 0.4, 0.6, 0.1]

Combined visibility: [1.0, 0.72, 0.72, 0.12, 0.42, 0.02]

Weighted sum:
  = 2.5×1.0 + 3.0×0.72 + 8.0×0.72 + 7.5×0.12 + 6.0×0.42 + 9.0×0.02
  = 2.5 + 2.16 + 5.76 + 0.9 + 2.52 + 0.18
  = 14.02

Total visibility: 1.0 + 0.72 + 0.72 + 0.12 + 0.42 + 0.02 = 3.0

Final distance = 14.02 / 3.0 = 4.67
```

### 4. **Feature Normalization**

**Important:** Your config has `normalize_feature: True`

This means features are L2-normalized before distance computation:

```python
# Each embedding vector is normalized to unit length
embedding_normalized = embedding / ||embedding||_2

# This constrains distances to a specific range:
# - Min distance: 0 (identical vectors)
# - Max distance: 2 (opposite vectors)
# - Typical range for similar persons: 0-1
# - Typical range for different persons: 1-2
```

**Without normalization:** Distances could be any positive value (0 to infinity)
**With normalization:** Distances are bounded [0, 2]

### 5. **Threshold Comparison**

Finally, the computed distance is compared to your threshold:

```python
# From video_reid_inference.py:479-484
if dist < args.threshold:
    label = f"ID {pid} (d={dist:.2f})"
    color = (0, 255, 0)  # Green - MATCH
else:
    label = f"Unknown (d={dist:.2f})"
    color = (0, 0, 255)  # Red - NO MATCH
```

**Your current threshold: 15.0**

---

## Why Distance = 9 Shows as ID

With normalized features, distances are typically in range [0, 2]:
- Distance 9.0 is **extremely unusual** and suggests something might be wrong
- However, 9.0 < 15.0, so it correctly shows as a match

**Possible explanations for d=9:**
1. Features might not be normalized in inference (check if `normalize_feature` is being applied)
2. The visibility weighting might be creating larger-than-expected distances
3. Distances might be summed instead of averaged somewhere

---

## Recommended Threshold Values

Based on normalized features (typical ReID behavior):

| Threshold | Behavior |
|-----------|----------|
| **0.5** | Very strict - only near-identical matches |
| **1.0** | Moderate - same person from different angles |
| **1.5** | Loose - allows more variation |
| **2.0** | Very loose - maximum possible distance |
| **5.0+** | Too large - will match everyone |

**For your case with d=9:**
- If you want stricter matching, try `--threshold 5.0` or `--threshold 7.0`
- If features are normalized properly, try `--threshold 1.0` or `--threshold 1.5`

---

## Debugging Distance Values

To understand what distances you're getting, check the console output:

```bash
# Run inference and look for distance values
./test_embedding_inference.sh 2>&1 | grep "d="

# Or add print statements in video_reid_inference.py around line 431:
print(f"Debug: Query {i}, Top match: PID={best_pid}, Distance={best_dist:.4f}")
```

**Expected behavior with normalized features:**
- Same person: d = 0.3 - 0.8
- Different person: d = 1.2 - 1.8
- Very different person: d = 1.8 - 2.0

**If you're seeing d=9:**
- This suggests features are not normalized
- Or the distance calculation is being scaled somehow

---

## Configuration Summary

**Your current setup:**
```yaml
test:
  normalize_feature: True          # Features should be normalized
  dist_metric: 'euclidean'         # Euclidean distance
  part_based:
    dist_combine_strat: 'mean'     # Visibility-weighted mean

model:
  bpbreid:
    test_embeddings: ['bn_foreg', 'parts']  # 6 parts total
```

**Inference threshold:**
```bash
--threshold 15.0  # Current value
```

---

## How to Adjust Threshold

1. **In test scripts** ([test_embedding_inference.sh](test_embedding_inference.sh)):
   ```bash
   # Change line 45, 75, 105:
   --threshold 5.0  # or whatever value you want
   ```

2. **In direct inference**:
   ```bash
   python3 video_reid_inference.py \
       --config configs/test_reid.yaml \
       --gallery-dir gallery_bank \
       --video input.mp4 \
       --threshold 1.0  # Your desired threshold
   ```

3. **For RTSP streams** ([video_reid_rtsp.py](video_reid_rtsp.py)):
   ```bash
   python3 video_reid_rtsp.py \
       --config configs/test_reid.yaml \
       --gallery-dir gallery_bank \
       --rtsp rtsp://your-stream \
       --threshold 1.0
   ```

---

## Summary

**The threshold works correctly** - it compares the computed distance against your specified value.

**The real question is:** What should the threshold be?
- With normalized features: typically 0.5 - 1.5
- With unnormalized features: depends on your data (could be 5.0 - 20.0)

**To diagnose:**
1. Check actual distance values in your output
2. Verify if features are being normalized
3. Set threshold based on observed distances between same/different persons
