"""
3-batch voting system for exercise classification.

Collects predictions from 3 consecutive 64-frame batches
and performs majority voting to determine final exercise.
"""
import time
import threading
from collections import defaultdict, Counter
from typing import Dict, List, Optional
from .models import BatchPrediction, VotingResult, TrackKey


class VotingSystem:
    """
    Implements 3-batch voting for exercise classification.

    Voting rules:
    - Requires 3 consecutive batch predictions
    - Needs 2/3 agreement (2 or 3 votes) for consensus
    - Combines frames from all 3 batches (192 frames total)
    """

    def __init__(self, confidence_threshold: float = 0.3, batches_required: int = 3):
        """
        Initialize voting system.

        Args:
            confidence_threshold: Minimum confidence per batch
            batches_required: Number of batches required for voting (default: 3)
        """
        self.confidence_threshold = confidence_threshold
        self.batches_required = batches_required

        # Per-track batch collection
        self.batch_predictions: Dict[TrackKey, List[BatchPrediction]] = defaultdict(list)
        self.voting_results: Dict[TrackKey, VotingResult] = {}

        self.lock = threading.Lock()

        # Statistics
        self.total_batches_added = 0
        self.total_votes_completed = 0
        self.failed_votes = 0

    def add_prediction(self, cam_id: str, track_id: int,
                      prediction: BatchPrediction) -> Optional[VotingResult]:
        """
        Add batch prediction and check if voting can complete.

        Args:
            cam_id: Camera identifier
            track_id: Person track ID
            prediction: Batch prediction to add

        Returns:
            VotingResult if voting completed, None otherwise
        """
        key = (cam_id, track_id)

        with self.lock:
            # Add prediction
            self.batch_predictions[key].append(prediction)
            self.total_batches_added += 1

            # Check if we have enough batches
            predictions = self.batch_predictions[key]

            if len(predictions) >= self.batches_required:
                # Take first N batches
                voting_set = predictions[:self.batches_required]

                # Perform voting
                result = self.perform_voting(voting_set)

                if result:
                    # Store result
                    self.voting_results[key] = result
                    self.total_votes_completed += 1

                    # Remove used batches
                    self.batch_predictions[key] = predictions[self.batches_required:]

                    return result
                else:
                    # Voting failed, remove first batch and try again later
                    self.batch_predictions[key] = predictions[1:]
                    self.failed_votes += 1

        return None

    def perform_voting(self, predictions: List[BatchPrediction]) -> Optional[VotingResult]:
        """
        Perform voting on batch predictions.

        Args:
            predictions: List of batch predictions (should be 3)

        Returns:
            VotingResult if consensus reached, None otherwise
        """
        # Count votes
        votes = Counter([pred.exercise for pred in predictions])

        # Get most common
        most_common = votes.most_common(1)[0]
        exercise_name, vote_count = most_common

        # Need at least 2/3 agreement
        if vote_count < 2:
            return None

        # Combine frames from all batches
        all_frames = []
        batch_ids = []
        confidences = []

        for pred in predictions:
            all_frames.extend(pred.frames)
            batch_ids.append(pred.batch_id)
            confidences.append(pred.confidence)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)

        return VotingResult(
            exercise=exercise_name,
            confidence=avg_confidence,
            frames=all_frames,
            vote_counts=dict(votes),
            batch_ids=batch_ids,
            timestamp=time.time()
        )

    def get_pending_results(self) -> Dict[TrackKey, VotingResult]:
        """
        Get and clear all pending voting results.

        Returns:
            Dictionary of {(cam_id, track_id): VotingResult}
        """
        with self.lock:
            results = self.voting_results.copy()
            self.voting_results.clear()
            return results

    def get_stats(self) -> dict:
        """Get voting statistics."""
        with self.lock:
            return {
                'total_batches_added': self.total_batches_added,
                'total_votes_completed': self.total_votes_completed,
                'failed_votes': self.failed_votes,
                'pending_results': len(self.voting_results),
                'tracks_with_pending_batches': len(self.batch_predictions)
            }
