"""
Weight Recognition Module

Modular weight detection and recognition system:
- YOLO-based detection
- RepVGG embedding recognition
- Gallery search matching
"""

from .weight_service import WeightService
from .weight_detector import WeightDetector
from .weight_recognizer import WeightRecognizer

__all__ = [
    'WeightService',
    'WeightDetector',
    'WeightRecognizer',
]
