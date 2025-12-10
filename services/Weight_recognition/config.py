"""
Configuration for Weight Recognition module.
Centralized configuration to avoid hardcoded values.
"""

# Model paths
WEIGHTS_YOLO = "models/best_10.pt"
EMBEDDINGS_JSON = "models/new_embed2.json"
EMB_MODEL_NAME = "repvgg_a2"

# Detection parameters
IMGSZ = 640
CONF = 0.5
IOU = 0.45
MAX_DET = 6

# Recognition thresholds
DIST_ACCEPT = 0.95
DIST_MARGIN = 0.05

# Crop parameters
PAD_BASE = 0.05
PAD_MIN = 0.02
PAD_MAX = 0.08
MULTICROP_PAD_FACTORS = [0.6, 1.0, 1.35]

# Foreground masking
SAT_S = 0.28
SAT_V = 0.20
SAT_ERODE = 1
SAT_DILATE = 1
