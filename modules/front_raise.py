# modules/front_raise.py

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
        'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
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

class FrontRaise:
    def __init__(self):
        self.stages = {'left': 'down', 'right': 'down'}
        self.rep_counters = {'left': 0, 'right': 0}
        self.feedback_log = {'left': [], 'right': []}
        self.frame_info = {'left': {}, 'right': {}}
        self.REP_COUNT_UP_THRESHOLD, self.REP_COUNT_DOWN_THRESHOLD = 65, 35
        self.FORM_UP_THRESHOLD, self.FORM_DOWN_THRESHOLD = 80, 25
        self.SWAY_THRESHOLD_X, self.KNEE_ANGLE_THRESHOLD = 25, 165
        self.SHOULDER_SHRUG_THRESHOLD, self.ARM_STRAIGHT_THRESHOLD = 0.25, 155
        self.OVER_SWING_THRESHOLD, self.WRIST_Y_ALIGNMENT_TOLERANCE = 105, 0.1

    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="FrontRaise",
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
            
        baseline_hip_x = (df.iloc[0]['left_hip_x'] + df.iloc[0]['right_hip_x']) / 2
        current_rep_feedback = {'left': [], 'right': []}
        raise_angle_at_peak = {'left': 0, 'right': 0}

        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            
            current_hip_x = (row['left_hip_x'] + row['right_hip_x']) / 2
            if abs(current_hip_x - baseline_hip_x) > self.SWAY_THRESHOLD_X:
                msg = "Body swaying. Brace your core and use less momentum."
                current_rep_feedback['left'].append(msg)
                current_rep_feedback['right'].append(msg)

            torso_height = abs(row['left_shoulder_y'] - row['left_hip_y'])
            if torso_height > 0:
                y_diff_norm = abs(row['left_wrist_y'] - row['right_wrist_y']) / torso_height
                if y_diff_norm > self.WRIST_Y_ALIGNMENT_TOLERANCE:
                    msg = "Uneven lifting. Raise both arms at the same time."
                    current_rep_feedback['left'].append(msg)
                    current_rep_feedback['right'].append(msg)

            for side in ['left', 'right']:
                if row[f'{side}_knee_angle'] < self.KNEE_ANGLE_THRESHOLD:
                    current_rep_feedback[side].append(f"Using leg momentum. Keep {side} leg straight.")
                if row[f'{side}_elbow_angle'] < self.ARM_STRAIGHT_THRESHOLD:
                    current_rep_feedback[side].append(f"Bending {side} arm. Keep it straighter.")
                shoulder_width = abs(row['left_shoulder_x'] - row['right_shoulder_x'])
                if shoulder_width > 0:
                    shrug_ratio = abs(row[f'{side}_ear_y'] - row[f'{side}_shoulder_y']) / shoulder_width
                    if shrug_ratio < self.SHOULDER_SHRUG_THRESHOLD:
                         current_rep_feedback[side].append(f"{side.capitalize()} shoulder shrugging. Focus on your deltoids.")

                raise_angle = row[f'{side}_raise_angle']

                if raise_angle > self.OVER_SWING_THRESHOLD:
                    current_rep_feedback[side].append("Lifting too high. Stop at shoulder level.")

                if raise_angle > self.REP_COUNT_UP_THRESHOLD and self.stages[side] == 'down':
                    if local_idx > 0 and df.iloc[local_idx-1][f'{side}_raise_angle'] > self.FORM_DOWN_THRESHOLD:
                        current_rep_feedback[side].append("Incomplete bottom range of motion. Lower arms fully.")
                    self.stages[side] = 'up'
                    self.rep_counters[side] += 1
                    self.frame_info[side][f"rep_{self.rep_counters[side]}"] = int(frame_num)

                if raise_angle < self.REP_COUNT_DOWN_THRESHOLD and self.stages[side] == 'up':
                    if raise_angle_at_peak[side] < self.FORM_UP_THRESHOLD:
                        current_rep_feedback[side].append("Incomplete top range of motion. Lift higher.")
                    feedback_to_add = current_rep_feedback[side] if current_rep_feedback[side] else ["Good Rep!"]
                    self.feedback_log[side].append(_unique_preserve(feedback_to_add))
                    current_rep_feedback[side].clear()
                    self.stages[side] = "down"
                    raise_angle_at_peak[side] = 0
                
                if self.stages[side] == 'up':
                    raise_angle_at_peak[side] = max(raise_angle_at_peak[side], raise_angle)

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
# "Body swaying. Brace your core and use less momentum."
#   - Trigger: The horizontal midpoint of the hips deviates more than SWAY_THRESHOLD_X from its starting position.
#   - Why it's bad: Uses back and hip momentum instead of isolating the shoulders.
#
# "Using leg momentum. Keep {side} leg straight."
#   - Trigger: The knee angle bends to less than KNEE_ANGLE_THRESHOLD.
#   - Why it's bad: Uses leg drive to lift the weight, reducing focus on the shoulders.
#
# ERROR FEEDBACK (ARMS & SHOULDERS):
# "Bending {side} arm. Keep it straighter."
#   - Trigger: The elbow angle becomes less than ARM_STRAIGHT_THRESHOLD.
#   - Why it's bad: Engages the bicep and reduces focus on the front deltoids.
#
# "{Side} shoulder shrugging. Focus on using your deltoids."
#   - Trigger: The vertical distance between the ear and shoulder becomes too small relative to the shoulder width.
#   - Why it's bad: Lifts the weight with the trapezius muscles instead of the shoulders.
#
# "Uneven lifting. Raise both arms at the same time."
#   - Trigger: The normalized vertical distance between the left and right wrists exceeds WRIST_Y_ALIGNMENT_TOLERANCE.
#   - Why it's bad: Can lead to muscle imbalances and indicates a lack of control.
#
# "Incomplete top range of motion. Lift higher."
#   - Trigger: A rep is completed, but the peak raise angle achieved was less than the strict FORM_UP_THRESHOLD.
#   - Why it's bad: Fails to fully contract the front deltoid, making the exercise less effective.
#
# "Incomplete bottom range of motion. Lower arms fully."
#   - Trigger: A new rep is started, but the raise angle in the previous frame was still greater than the strict FORM_DOWN_THRESHOLD.
#   - Why it's bad: This is a "partial rep" that reduces time under tension and overall effectiveness.
#
# "Lifting too high. Stop at shoulder level."
#   - Trigger: The raise angle (hip-shoulder-wrist) exceeds the OVER_SWING_THRESHOLD.
#   - Why it's bad: Can place unnecessary stress on the shoulder joint and rotator cuff.
#