"""
YOLO mask transforms for 17 COCO keypoints
Maps YOLO's 17 keypoints to body part groups compatible with BPBreID
"""
from .mask_transform import MaskGroupingTransform

# YOLO COCO 17 keypoints mapping
YOLO_COCO_PARTS_MAP = {
    # Keypoints (0-16)
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


class CombineYOLOIntoFiveVerticalParts(MaskGroupingTransform):
    """
    Group YOLO's 17 keypoints into 5 vertical body parts
    Compatible with BPBreID's five_v preprocessing
    """
    parts_grouping = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "upper_arms_torso_mask": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        "lower_arms_torso_mask": ["left_wrist", "right_wrist", "left_hip", "right_hip"],
        "legs_mask": ["left_hip", "right_hip", "left_knee", "right_knee"],
        "feet_mask": ["left_ankle", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, YOLO_COCO_PARTS_MAP)
