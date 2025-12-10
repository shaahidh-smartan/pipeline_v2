"""
Person Re-Identification Module.
Core ReID processing and caching functions.
"""

from .reid_processing import process_frame_for_reid, prepare_detections_for_tracker
from .caching import (
    get_cached_person,
    cache_person_identification,
    update_cache_timestamp,
    cache_master_list,
    cleanup_expired_cache_entries
)

__all__ = [
    'process_frame_for_reid',
    'prepare_detections_for_tracker',
    'get_cached_person',
    'cache_person_identification',
    'update_cache_timestamp',
    'cache_master_list',
    'cleanup_expired_cache_entries',
]
