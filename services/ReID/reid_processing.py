"""
ReID Processing Functions.
Core person detection, tracking, and identification logic.
"""
import time
import torch
import traceback
import numpy as np


def prepare_detections_for_tracker(person_boxes_track):
    if not person_boxes_track:
        return torch.tensor([], dtype=torch.float32).reshape(0, 6)

    # Convert to numpy array first
    detections_array = np.array(person_boxes_track)

    # Check if we have the right shape
    if detections_array.ndim != 2 or detections_array.shape[1] != 5:
        return torch.tensor([], dtype=torch.float32).reshape(0, 6)

    # Convert to format expected by BYTETracker [x1, y1, x2, y2, score, class]
    detections_list = []
    for box in person_boxes_track:
        if len(box) >= 5:
            x1, y1, x2, y2, score = box[:5]
            w = x2 - x1
            h = y2 - y1

            # Filter by area and aspect ratio
            if w > 0 and h > 0 and w * h > 100 and w / h <= 2.0:
                detection = [float(x1), float(y1), float(x2), float(y2), float(score), 1.0]
                detections_list.append(detection)

    if not detections_list:
        return torch.tensor([], dtype=torch.float32).reshape(0, 6)

    return torch.tensor(detections_list, dtype=torch.float32)


def process_frame_for_reid(frame, camera_id, person_detector, person_embedder, similarity_search,
                           trackers, last_detection_time, track_id_to_person, similarity_threshold,
                           get_cached_person_func, cache_person_identification_func,
                           update_cache_timestamp_func, cache_master_list_func):
    try:
        # Detect persons using person detector utility
        person_detections, person_boxes_track = person_detector.detect_persons(frame)

        now = time.time()
        # Extract camera number from camera_id (e.g., 'cam_1' -> 1)
        sample_id = int(camera_id.split('_')[1]) - 1

        if len(person_boxes_track) > 0:
            last_detection_time[sample_id] = now
        else:
            last_seen = last_detection_time.get(sample_id, now)

            if now - last_seen > 0.1:
                # print(f"[DEBUG] No person detected for 0.1s on camera {sample_id}, clearing tracking history")

                tracker = trackers[camera_id]
                empty_outputs = torch.tensor([], dtype=torch.float32).reshape(0, 6)

                if torch.cuda.is_available():
                    empty_outputs = empty_outputs.cuda()

                try:
                    online_targets = tracker.update(
                        empty_outputs,
                        [frame.shape[0], frame.shape[1]],
                        (frame.shape[0], frame.shape[1])
                    )

                    # Clear cache for this camera when tracking is reset
                    keys_to_remove = [key for key in track_id_to_person.keys() if key[0] == camera_id]
                    for key in keys_to_remove:
                        del track_id_to_person[key]

                except Exception as e:
                    print(f"Error updating tracker with empty values: {e}")

                last_detection_time[sample_id] = now

        if not person_detections or not person_boxes_track:
            return []

        target_box = []
        reid_results = []
        similarity_check = 0

        # Prepare detections for tracking
        outputs = prepare_detections_for_tracker(person_boxes_track)

        if torch.cuda.is_available() and hasattr(trackers[camera_id], 'args'):
            device = getattr(trackers[camera_id].args, 'device', 'cpu')
            if hasattr(device, 'type'):
                device_str = device.type
            else:
                device_str = str(device)

            if 'cuda' in device_str:
                outputs = outputs.cuda()

        try:
            if outputs is not None and len(outputs) > 0:
                online_targets = trackers[camera_id].update(
                    outputs,
                    [frame.shape[0], frame.shape[1]],
                    (frame.shape[0], frame.shape[1])
                )

                for target in online_targets:
                    target_box.append([target.track_id, target.tlwh[0], target.tlwh[1],
                                    target.tlwh[2], target.tlwh[3], target.track_id, camera_id])

            else:
                online_targets = []
        except Exception as e:
            traceback.print_exc()
            return []

        # Process each target for ReID with database similarity search and caching
        for target in target_box:
            track_id, x, y, w, h, _, cam_id = target

            # Check cache first
            cached_person = get_cached_person_func(cam_id, track_id)

            if cached_person:
                # Use cached result
                person_name = cached_person['name']
                similarity = cached_person['confidence']

                # Update cache timestamp to keep it alive
                update_cache_timestamp_func(cam_id, track_id)

                # Convert coordinates
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                bbox = [x1, y1, x2, y2]
                cache_master_list_func(cached_person['name'], cam_id, track_id)
                # Create match object for consistency
                match = {
                    'person_name': person_name,
                    'similarity': similarity,
                    'source_camera': 'cached'
                } if person_name != "Unknown" else None

                reid_results.append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'match': match,
                    'similarity': similarity,
                    'person_name': person_name,
                    'camera_id': cam_id,
                    'cached': True  # Flag to indicate this was from cache
                })

            else:
                # No cache hit, perform database ReID
                # Convert from TLWH to TLBR
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                # Get person crop using person detector utility
                person_crop = person_detector.get_person_crop(frame, [x1, y1, x2, y2], padding=10)

                if person_crop is None:
                    print(f"Empty crop for track_id {track_id}, skipping")
                    continue

                try:
                    # Extract embedding using person embedder
                    embedding = person_embedder.extract_embedding(person_crop)

                    if embedding is not None:
                        # Database similarity search using cosine similarity
                        match, similarity = similarity_search.find_person_match_cosine(
                            embedding, threshold=similarity_threshold
                        )
                        similarity_check += 1

                        if match is not None:
                            person_name = match.get('person_name')
                            # Try to cache this result if confidence is high
                            cache_person_identification_func(cam_id, track_id, person_name, similarity)
                        else:
                            person_name = "Unknown"
                            # Don't cache "Unknown" - it should be re-checked each time
                            # since new people could be added to the database

                        # Store detailed results
                        reid_results.append({
                            'track_id': track_id,
                            'bbox': [x1, y1, x2, y2],
                            'match': match,
                            'similarity': similarity,
                            'person_name': person_name,
                            'camera_id': cam_id,
                            'cached': False  # Flag to indicate this was freshly computed
                        })
                    else:
                        print(f"Track ID {track_id}: Failed to extract embedding")

                except Exception as e:
                    print(f"Error processing track_id {track_id}: {e}")
                    continue
        # print("==========================End of reid========================")
        return reid_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []
