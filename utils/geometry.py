"""
Geometry utility functions for bounding boxes, IoU calculations, and display operations.
Consolidated from main.py, camera.py, and services/person_reid_service.py
"""
import numpy as np
import math


def clamp_bbox(bbox, h, w):
    """
    Clamp bounding box coordinates to image boundaries.

    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        h: Image height
        w: Image width

    Returns:
        Clamped bbox or None if invalid
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w,     int(x2)))
    y2 = max(0, min(h,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_and_clamp_bbox(bbox, h, w, px=0.35, py=0.25):
    """
    Expand bbox by percentage for better coverage and clamp to image boundaries.

    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        h: Image height
        w: Image width
        px: Horizontal expansion percentage
        py: Vertical expansion percentage

    Returns:
        Expanded and clamped bbox or None if invalid
    """
    x1, y1, x2, y2 = bbox
    bw, bh = (x2 - x1), (y2 - y1)
    dx, dy = int(px * bw), int(py * bh)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w-1, x2 + dx)
    ny2 = min(h-1, y2 + dy)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: Tuple of (x1, y1, x2, y2)
        box2: Tuple of (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def point_in_box(point, box):
    """
    Check if point (x,y) is inside bounding box [x1,y1,x2,y2].

    Args:
        point: Tuple of (x, y)
        box: Tuple of (x1, y1, x2, y2)

    Returns:
        Boolean indicating if point is inside box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_display_dimensions(original_width, original_height, target_width, target_height):
    """
    Calculate dimensions to maintain aspect ratio within target dimensions.

    Args:
        original_width: Original image width
        original_height: Original image height
        target_width: Target display width
        target_height: Target display height

    Returns:
        Tuple of (new_width, new_height, pad_left, pad_top, pad_right, pad_bottom)
    """
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    if original_aspect > target_aspect:
        new_width = target_width
        new_height = int(target_width / original_aspect)
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        return new_width, new_height, 0, pad_top, 0, pad_bottom
    else:
        new_height = target_height
        new_width = int(target_height * original_aspect)
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        return new_width, new_height, pad_left, 0, pad_right, 0


def calculate_angle(p1, p2, p3):
    """
    Calculate angle between three points using the law of cosines.

    Args:
        p1: First point (x, y)
        p2: Vertex point (x, y)
        p3: Third point (x, y)

    Returns:
        Angle in degrees
    """
    try:
        # Convert points to numpy arrays
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0

        # Calculate angle using dot product formula
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1, 1)

        # Convert from radians to degrees
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    except Exception:
        return 0


def is_face_inside_person(face_box, person_box, margin=10):
    """
    Check if face bounding box is inside person bounding box.

    Args:
        face_box: Face bbox (x1, y1, x2, y2)
        person_box: Person bbox (x1, y1, x2, y2)
        margin: Additional margin for person box

    Returns:
        Boolean indicating if face is inside person box
    """
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box

    px1 -= margin
    py1 -= margin
    px2 += margin
    py2 += margin

    return fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2


def calculate_overlap(box1, box2):
    """
    Calculate IoU overlap between two bounding boxes.
    (Alias for compute_iou for backward compatibility)

    Args:
        box1: First bbox (x1, y1, x2, y2)
        box2: Second bbox (x1, y1, x2, y2)

    Returns:
        IoU overlap value
    """
    return compute_iou(box1, box2)