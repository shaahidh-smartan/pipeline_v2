"""
Global exercise tracker for maintaining exercise history per track.

This module manages a master list of all detected exercises for each
person (track_id), including RepNet processing status.
"""
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from .models import ExerciseEntry, TrackKey


class GlobalExerciseTracker:
    """
    Global tracker that maintains a master list of all exercises per track id.

    Thread-safe tracker that:
    - Stores complete exercise history per (camera_id, track_id)
    - Manages RepNet processing queue
    - Provides statistics and summaries
    """

    def __init__(self):
        """Initialize empty tracker with threading lock."""
        self._master_exercise_list: Dict[TrackKey, List[ExerciseEntry]] = defaultdict(list)
        self._repnet_sent_index: Dict[TrackKey, int] = defaultdict(int)
        self._lock = threading.Lock()

        # Statistics
        self.total_exercises_added = 0
        self.total_repnet_processed = 0

    def add_exercise_entry(self, cam_id: str, track_id: int,
                          exercise_entry: ExerciseEntry) -> int:
        """
        Add a new exercise entry to the master list.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID
            exercise_entry: Exercise data to add

        Returns:
            int: Index of added entry in the list
        """
        key = (cam_id, track_id)

        with self._lock:
            self._master_exercise_list[key].append(exercise_entry)
            self.total_exercises_added += 1
            entry_index = len(self._master_exercise_list[key]) - 1

            print(f"[MASTER_LIST] Added {cam_id}:{track_id} #{entry_index}: "
                  f"{exercise_entry.exercise} ({exercise_entry.confidence:.3f}) - "
                  f"{len(exercise_entry.frames)} frames")

            return entry_index

    def get_entry_for_repnet(self, cam_id: str, track_id: int) -> Optional[Tuple[ExerciseEntry, int]]:
        """
        Get the next unprocessed exercise entry for RepNet.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID

        Returns:
            Tuple of (exercise_entry, index) or None if no unprocessed entries
        """
        key = (cam_id, track_id)

        with self._lock:
            exercise_list = self._master_exercise_list.get(key, [])
            last_sent_index = self._repnet_sent_index.get(key, -1)

            # Find next unprocessed entry
            for i in range(last_sent_index + 1, len(exercise_list)):
                entry = exercise_list[i]
                if not entry.processed_by_repnet:
                    self._repnet_sent_index[key] = i
                    print(f"[REPNET_SEND] {cam_id}:{track_id} - "
                          f"Sending exercise #{i}: {entry.exercise} to RepNet")
                    return entry, i

            return None

    def mark_repnet_processed(self, cam_id: str, track_id: int,
                             entry_index: int, reps: int,
                             rep_conf: float, stride: int) -> None:
        """
        Update entry with RepNet results.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID
            entry_index: Index of entry to update
            reps: Number of repetitions detected
            rep_conf: Confidence score
            stride: Stride used in RepNet
        """
        key = (cam_id, track_id)

        with self._lock:
            exercise_list = self._master_exercise_list.get(key, [])

            if 0 <= entry_index < len(exercise_list):
                entry = exercise_list[entry_index]
                entry.processed_by_repnet = True
                entry.reps = reps
                entry.rep_conf = rep_conf
                entry.repnet_stride = stride
                self.total_repnet_processed += 1

                print(f"[REPNET_DONE] {cam_id}:{track_id} #{entry_index}: "
                      f"{entry.exercise} -> {reps} reps (conf: {rep_conf:.3f})")

    def get_track_history(self, cam_id: str, track_id: int) -> List[ExerciseEntry]:
        """
        Get complete exercise history for a track.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID

        Returns:
            Copy of all exercise entries for the track
        """
        key = (cam_id, track_id)
        with self._lock:
            return self._master_exercise_list.get(key, []).copy()

    def get_track_summary(self, cam_id: str, track_id: int) -> dict:
        """
        Get summary statistics for a track.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID

        Returns:
            Dictionary with track statistics
        """
        key = (cam_id, track_id)

        with self._lock:
            exercise_list = self._master_exercise_list.get(key, [])
            last_sent_index = self._repnet_sent_index.get(key, -1)
            processed_count = sum(1 for e in exercise_list if e.processed_by_repnet)
            pending_count = len(exercise_list) - last_sent_index - 1

            return {
                'total_exercises': len(exercise_list),
                'processed_by_repnet': processed_count,
                'last_sent_index': last_sent_index,
                'pending_for_repnet': pending_count,
                'exercises': [e.exercise for e in exercise_list]
            }

    def get_global_stats(self) -> dict:
        """
        Get global statistics across all tracks.

        Returns:
            Dictionary with global statistics
        """
        with self._lock:
            active_tracks = len(self._master_exercise_list)
            total_exercises = sum(len(ex) for ex in self._master_exercise_list.values())

            return {
                'active_tracks': active_tracks,
                'total_exercises_in_master_list': total_exercises,
                'total_exercises_added': self.total_exercises_added,
                'total_repnet_processed': self.total_repnet_processed,
                'tracks': {f"{k[0]}:{k[1]}": len(v)
                          for k, v in self._master_exercise_list.items()}
            }


# Global singleton instance
global_exercise_tracker = GlobalExerciseTracker()
