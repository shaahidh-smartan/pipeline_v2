# modules/db_reverse_flys.py

import numpy as np
import json
from datetime import datetime
from decimal import Decimal

# --- Helper Functions (Standardized) ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def _unique_preserve(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, str, int, float)): return obj
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.ndarray,)):   return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, Decimal):         return float(obj)
    if isinstance(obj, datetime):        return obj.isoformat()
    if isinstance(obj, dict):            return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [_json_safe(x) for x in obj]
    return str(obj)

def calculate_angle(p1, p2, p3):
    if any(coord == (0, 0) for coord in [p1, p2, p3]): return 0
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1, v2 = p1 - p2, p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0: return 0
    cosine_angle = np.clip(dot_product / magnitude, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def preprocess_data_for_analysis(df):
    all_kps = []
    for _, row in df.iterrows():
        try:
            kps_list = json.loads(row['keypoints_json'])['kps']
            kps_dict = {name: (kp[0], kp[1]) for name, kp in zip(KEYPOINT_NAMES, kps_list)}
        except (json.JSONDecodeError, KeyError, TypeError):
            kps_dict = {name: (0, 0) for name in KEYPOINT_NAMES}
        all_kps.append(kps_dict)
    
    for name in KEYPOINT_NAMES:
        df[f'{name}_x'] = [kps.get(name, (0, 0))[0] for kps in all_kps]
        df[f'{name}_y'] = [kps.get(name, (0, 0))[1] for kps in all_kps]

    angles_to_calculate = {
        'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
        'left_raise_angle': ('left_hip', 'left_shoulder', 'left_wrist'),
        'right_raise_angle': ('right_hip', 'right_shoulder', 'right_wrist'),
    }

    for angle_name, points in angles_to_calculate.items():
        p1, p2, p3 = points
        df[angle_name] = df.apply(
            lambda row: calculate_angle(
                (row[f'{p1}_x'], row[f'{p1}_y']),
                (row[f'{p2}_x'], row[f'{p2}_y']),
                (row[f'{p3}_x'], row[f'{p3}_y'])
            ), axis=1
        )
    return df

class DB_Reverse_Flys:
    def __init__(self):
        self.stages = {'left': 'down', 'right': 'down'}
        self.rep_counters = {'left': 0, 'right': 0}
        self.feedback_log = {'left': [], 'right': []}
        self.frame_info = {'left': {}, 'right': {}}
        self.REP_COUNT_UP_THRESHOLD, self.REP_COUNT_DOWN_THRESHOLD = 30, 20
        self.FORM_UP_THRESHOLD, self.FORM_DOWN_THRESHOLD = 40, 15
        self.BENT_OVER_MIN_ANGLE, self.BENT_OVER_MAX_ANGLE = 70, 110
        self.ELBOW_MIN_ANGLE, self.ELBOW_MAX_ANGLE = 145, 175
        self.WRIST_Y_ALIGNMENT_TOLERANCE = 0.15

    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="DB_Reverse_Flys",
                      total_frames_analyzed=None,
                      analysis_timestamp=None,
                      metadata=None):
        if df.empty:
            return {
                "total_reps_computed": 0, "feedback": {}, "frame_info": {}, "exercise_name": exercise_name,
                "analyzer_used": analyzer_used, "total_frames_analyzed": int(total_frames_analyzed or 0),
                "analysis_timestamp": analysis_timestamp, "metadata": _json_safe(metadata or {})
            }
            
        df = preprocess_data_for_analysis(df)
            
        current_rep_feedback = {'left': [], 'right': []}
        fly_angle_at_peak = {'left': 0, 'right': 0}

        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            
            avg_hip_angle = (row['left_hip_angle'] + row['right_hip_angle']) / 2
            if not self.BENT_OVER_MIN_ANGLE < avg_hip_angle < self.BENT_OVER_MAX_ANGLE:
                msg = "Unstable back. Maintain a consistent bent-over posture."
                current_rep_feedback['left'].append(msg)
                current_rep_feedback['right'].append(msg)
            
            torso_height = abs(row['left_shoulder_y'] - row['left_hip_y'])
            if torso_height > 0:
                y_diff_norm = abs(row['left_wrist_y'] - row['right_wrist_y']) / torso_height
                if y_diff_norm > self.WRIST_Y_ALIGNMENT_TOLERANCE:
                    msg = "Uneven arms. Move both dumbbells together."
                    current_rep_feedback['left'].append(msg)
                    current_rep_feedback['right'].append(msg)

            for side in ['left', 'right']:
                elbow_angle = row[f'{side}_elbow_angle']
                if not self.ELBOW_MIN_ANGLE < elbow_angle < self.ELBOW_MAX_ANGLE:
                    current_rep_feedback[side].append(f"Rowing with {side} arm. Keep a fixed bend in your elbow.")

                fly_angle = row[f'{side}_raise_angle']

                if fly_angle > self.REP_COUNT_UP_THRESHOLD and self.stages[side] == 'down':
                    self.stages[side] = 'up'
                    self.rep_counters[side] += 1
                    self.frame_info[side][f"rep_{self.rep_counters[side]}"] = int(frame_num)

                if fly_angle < self.REP_COUNT_DOWN_THRESHOLD and self.stages[side] == 'up':
                    if fly_angle_at_peak[side] < self.FORM_UP_THRESHOLD:
                        current_rep_feedback[side].append("Incomplete contraction. Squeeze your shoulder blades.")
                    feedback_to_add = current_rep_feedback[side] if current_rep_feedback[side] else ["Good Rep!"]
                    self.feedback_log[side].append(_unique_preserve(feedback_to_add))
                    current_rep_feedback[side].clear()
                    self.stages[side] = "down"
                    fly_angle_at_peak[side] = 0
                
                if self.stages[side] == 'up':
                    fly_angle_at_peak[side] = max(fly_angle_at_peak[side], fly_angle)

        for side in ['left', 'right']:
            if self.stages[side] == 'up' and len(self.feedback_log[side]) < self.rep_counters[side]:
                self.feedback_log[side].append(_unique_preserve(current_rep_feedback[side] if current_rep_feedback[side] else ["Good Rep!"]))

        total_reps = max(self.rep_counters['left'], self.rep_counters['right'])
        combined_feedback = {}
        for i in range(1, total_reps + 1):
            left_fb  = self.feedback_log['left'][i-1]  if i-1 < len(self.feedback_log['left'])  else []
            right_fb = self.feedback_log['right'][i-1] if i-1 < len(self.feedback_log['right']) else []
            merged = _unique_preserve(list(left_fb) + list(right_fb))
            if len(merged) > 1 and "Good Rep!" in merged:
                merged = [m for m in merged if m != "Good Rep!"]
            if not merged:
                merged = ["Good Rep!"]
            combined_feedback[f"rep_{i}"] = merged

        final_side = 'left' if self.rep_counters['left'] >= self.rep_counters['right'] else 'right'
        final_frame_info = {
            k: v for k, v in sorted(
                self.frame_info[final_side].items(),
                key=lambda kv: int(kv[0].split('_')[1])
            )
        }

        report = {
            "total_reps_computed": total_reps, "feedback": combined_feedback, "frame_info": final_frame_info,
            "exercise_name": exercise_name, "analyzer_used": analyzer_used,
            "total_frames_analyzed": int(total_frames_analyzed or len(df)),
            "analysis_timestamp": analysis_timestamp, "metadata": metadata or {}
        }
        
        return _json_safe(report)


# --- FEEDBACK AND TRIGGER CONDITIONS SUMMARY ---
#
# POSITIVE FEEDBACK:
# "Good Rep!"
#   - Trigger: A full repetition is successfully completed with no other error feedback logged.
#
# ERROR FEEDBACK (BODY):
# "Unstable back. Maintain a consistent bent-over posture."
#   - Trigger: The average hip angle (shoulder-hip-knee) goes outside the stable BENT_OVER range.
#   - Why it's bad: Indicates the user is standing up during the rep, which uses momentum and reduces focus on the target muscles.
#
# ERROR FEEDBACK (ARMS & SHOULDERS):
# "Rowing with {side} arm. Keep a fixed bend in your elbow."
#   - Trigger: The elbow angle goes outside the fixed ELBOW_MIN/MAX_ANGLE range.
#   - Why it's bad: This is the primary error. It turns the exercise into a row, engaging the lats and biceps instead of isolating the rear deltoids and upper back.
#
# "Uneven arms. Move both dumbbells together."
#   - Trigger: The normalized vertical distance between the left and right wrists exceeds WRIST_Y_ALIGNMENT_TOLERANCE.
#   - Why it's bad: Can lead to muscle imbalances and indicates a lack of control.
#
# "Incomplete contraction. Squeeze your shoulder blades."
#   - Trigger: A rep is completed, but the peak fly angle achieved (hip-shoulder-wrist) was less than the strict FORM_UP_THRESHOLD.
#   - Why it's bad: Fails to fully contract the rear deltoids and rhomboids at the peak of the movement.
#