"""
Person ReID Caching Functions.
Track-based caching for fast person identification retrieval.
"""
import time


def get_cached_person(track_id_to_person, cache_expiry_time, camera_id, track_id):
    cache_key = (camera_id, track_id)

    if cache_key in track_id_to_person:
        cached_entry = track_id_to_person[cache_key]
        current_time = time.time()

        # Check if cache entry hasn't expired
        if current_time - cached_entry['timestamp'] <= cache_expiry_time:
            return cached_entry
        else:
            # Cache expired, remove it
            print(f"[CACHE EXPIRED] Removing expired entry for track ID {track_id}")
            del track_id_to_person[cache_key]

    return None


def cache_person_identification(track_id_to_person, confidence_threshold_for_caching,
                                camera_id, track_id, person_name, confidence):
    if confidence >= confidence_threshold_for_caching:
        cache_key = (camera_id, track_id)
        track_id_to_person[cache_key] = {
            'name': person_name,
            'confidence': confidence,
            'timestamp': time.time()
        }
        print(f"[CACHE STORED] Track ID {track_id} -> {person_name} (confidence: {confidence:.3f})")
        return True
    return False


def update_cache_timestamp(track_id_to_person, camera_id, track_id):
    cache_key = (camera_id, track_id)
    if cache_key in track_id_to_person:
        track_id_to_person[cache_key]['timestamp'] = time.time()


def cache_master_list(master_person_list, logged_tracks, db_manager, person_name, camera_id, track_id):
    if person_name != "Unknown":
        # Create unique key for this track
        track_key = (camera_id, track_id, person_name)

        # Only save if we haven't saved this track before
        if track_key not in logged_tracks:
            # Add to master list
            entry = {
                'person_name': person_name,
                'camera_id': camera_id,
                'track_id': track_id,
                'timestamp': time.time()
            }
            master_person_list.append(entry)

            # Insert into database only
            try:
                db_success = db_manager.insert_person_reid_log(person_name, camera_id, track_id)

                if db_success:
                    print(f"[DB SAVED] {person_name} detected on {camera_id} with track {track_id}")
                else:
                    print(f"[DB FAILED] {person_name} detected on {camera_id} with track {track_id}")

                # Mark this track as saved
                logged_tracks.add(track_key)

            except Exception as e:
                print(f"Error saving to database: {e}")


def cleanup_expired_cache_entries(track_id_to_person, cache_expiry_time):
    current_time = time.time()
    expired_keys = [
        key for key, value in track_id_to_person.items()
        if current_time - value['timestamp'] > cache_expiry_time
    ]

    for key in expired_keys:
        del track_id_to_person[key]

    if expired_keys:
        print(f"[CACHE CLEANUP] Removed {len(expired_keys)} expired entries")
