"""
Data models for exercise recognition system.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class ExerciseEntry:
    """
    Single exercise entry with frames and metadata.

    This represents a completed exercise detection with voting results,
    frames, and optional RepNet/weight data.
    """
    exercise: str                      # Exercise name (e.g., "Bicep Curl")
    confidence: float                  # Voting confidence (0.0-1.0)
    frames: List                       # 192 frames (3 batches of 64)
    timestamp: float                   # Unix timestamp
    voting_cycle_id: str               # Unique ID for this voting cycle
    vote_counts: dict                  # Vote distribution {exercise: count}
    batches_used: int                  # Number of batches used (should be 3)
    batch_ids: list                    # Window IDs used in voting
    weight: str = "unknown"            # Detected weight (if any)
    weight_confidence: float = 0.0     # Weight detection confidence
    processed_by_repnet: bool = False  # RepNet processing status
    reps: Optional[int] = None         # Repetition count from RepNet
    rep_conf: Optional[float] = None   # RepNet confidence
    repnet_stride: Optional[int] = None # Stride used by RepNet


@dataclass
class EnhancedClipMeta:
    """
    Metadata for a single 64-frame window.

    Tracks the origin and sequence information for each frame window
    created by the frame buffer.
    """
    cam_id: str                        # Camera identifier
    track_id: int                      # Person track ID
    ts: float                          # Timestamp when window created
    window_start: int                  # Global start frame number
    window_end: int                    # Global end frame number
    seq_id_range: Tuple[int, int]      # (start_seq, end_seq) for this window


@dataclass
class FrameWindow:
    """
    A 64-frame window ready for processing.

    Output from FrameBuffer, input to voting system.
    """
    frames: List[np.ndarray]           # 64 frames (BGR)
    cam_id: str                        # Camera identifier
    track_id: int                      # Person track ID
    window_index: int                  # Global window counter
    seq_start: int                     # Starting sequence number
    seq_end: int                       # Ending sequence number
    timestamp: float                   # When window was created


@dataclass
class BatchPrediction:
    """
    Single batch prediction from inference.

    Represents one 64-frame batch's classification result.
    """
    exercise: str                      # Predicted exercise
    confidence: float                  # Model confidence
    frames: List[np.ndarray]           # 64 frames used
    batch_id: int                      # Window ID
    timestamp: float                   # Prediction timestamp


@dataclass
class VotingResult:
    """
    Final voting result from 3 batches.

    Output after 3-batch voting completes.
    """
    exercise: str                      # Consensus exercise
    confidence: float                  # Average confidence
    frames: List[np.ndarray]           # All 192 frames (3x64)
    vote_counts: Dict[str, int]        # Vote distribution
    batch_ids: List[int]               # Window IDs used
    timestamp: float                   # Voting completion time


# Type aliases for clarity
TrackKey = Tuple[str, int]  # (camera_id, track_id)
