"""
Exercise Recognition Module

Modular, model-agnostic exercise recognition system with:
- Frame buffering
- 3-batch voting
- Flexible model support
"""

from .models import ExerciseEntry, VotingResult, FrameWindow, BatchPrediction, TrackKey
from .exercise_tracker import GlobalExerciseTracker, global_exercise_tracker
from .main import ExerciseService

__all__ = [
    'ExerciseEntry',
    'VotingResult',
    'FrameWindow',
    'BatchPrediction',
    'TrackKey',
    'GlobalExerciseTracker',
    'global_exercise_tracker',
    'ExerciseService',
]
