"""
Frame buffer code file
this creates 64 frames non overlapping windows of incoming person.

for each camera-id and tracking-id:
    - buffers incoming frames
    - creates 64 frame window with 32 evenly spaced frame stride
"""

import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import queue
from .models import FrameWindow, TrackKey


class FrameBuffer:
    """
    Manages per-track frame buffering with emit cursor architecture.

    For each (camera_id, track_id):
    - Buffers incoming frames
    - Creates 64-frame windows with 32-frame stride
    - Prevents frame loss with emit cursor
    - Manages memory with rebasing
    """

    def __init__(self, window_size: int = 64, stride: int = 32):
        """
        Initialize frame buffer.

        Args:
            window_size: Number of frames per window (default: 64)
            stride: Stride between windows (default: 32)
        """
        self.window_size = window_size
        self.stride = stride

        # Per-track buffers
        self.stream_buf: Dict[TrackKey, List[np.ndarray]] = {}
        self.emit_idx: Dict[TrackKey, int] = {}
        self.seq_counters: Dict[TrackKey, int] = {}
        self.last_activity: Dict[TrackKey, float] = {}
        self.global_win_counter: Dict[TrackKey, int] = defaultdict(int)

        self.lock = threading.Lock()

        # Statistics
        self.windows_created = 0
        self.frames_received = 0

    def add_frame(self, cam_id: str, track_id: int,
                  frame: np.ndarray) -> List[FrameWindow]:
        """
        Add frame and return any completed windows.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID
            frame: BGR frame (person crop)

        Returns:
            List of completed FrameWindow objects (may be empty)
        """
        key = (cam_id, track_id)
        windows = []

        with self.lock:
            # Initialize if new track
            if key not in self.stream_buf:
                self.stream_buf[key] = []
                self.emit_idx[key] = 0
                self.seq_counters[key] = 0

            # Add frame to buffer
            self.stream_buf[key].append(frame)
            self.seq_counters[key] += 1
            self.last_activity[key] = time.time()
            self.frames_received += 1

            # Check if we can emit windows
            buf = self.stream_buf[key]
            emit_idx = self.emit_idx[key]

            while emit_idx + self.window_size <= len(buf):
                # Extract window
                window_frames = buf[emit_idx:emit_idx + self.window_size]

                # Create window object
                win_idx = self.global_win_counter[key]
                self.global_win_counter[key] += 1

                window = FrameWindow(
                    frames=window_frames.copy(),
                    cam_id=cam_id,
                    track_id=track_id,
                    window_index=win_idx,
                    seq_start=emit_idx,
                    seq_end=emit_idx + self.window_size,
                    timestamp=time.time()
                )

                windows.append(window)
                self.windows_created += 1

                # Move emit cursor
                emit_idx += self.stride
                self.emit_idx[key] = emit_idx

            # Rebase buffer if too large
            if len(buf) > 200:
                self.rebase_buffer(key)

        return windows

    def rebase_buffer(self, key: TrackKey):
        """Remove old frames to manage memory."""
        buf = self.stream_buf[key]
        emit_idx = self.emit_idx[key]

        if emit_idx > 100:
            # Keep last 100 frames before emit cursor
            keep_from = max(0, emit_idx - 100)
            self.stream_buf[key] = buf[keep_from:]
            self.emit_idx[key] = emit_idx - keep_from

    def cleanup_stale_tracks(self, max_age_seconds: float = 300):
        """Remove tracks with no recent activity."""
        current_time = time.time()

        with self.lock:
            stale_keys = [
                key for key, last_time in self.last_activity.items()
                if current_time - last_time > max_age_seconds
            ]

            for key in stale_keys:
                self.stream_buf.pop(key, None)
                self.emit_idx.pop(key, None)
                self.seq_counters.pop(key, None)
                self.last_activity.pop(key, None)

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self.lock:
            return {
                'active_tracks': len(self.stream_buf),
                'total_frames_received': self.frames_received,
                'total_windows_created': self.windows_created,
                'buffer_sizes': {f"{k[0]}:{k[1]}": len(v)
                                for k, v in self.stream_buf.items()}
            }
