# modules/bicep_curl.py
 
import numpy as np
import json
from datetime import datetime
from decimal import Decimal
 
# --- Helper Functions ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
 
def _unique_preserve(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out
 
def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    # numpy scalars/arrays
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.ndarray,)):   return [_json_safe(x) for x in obj.tolist()]
    # misc
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
 
class BicepCurls:
    def __init__(self):
        self.stages = {'left': 'down', 'right': 'down'}
        self.rep_counters = {'left': 0, 'right': 0}
        self.feedback_log = {'left': [], 'right': []}
        self.frame_info = {'left': {}, 'right': {}}
        self.REP_COUNT_UP_THRESHOLD, self.REP_COUNT_DOWN_THRESHOLD = 65, 130
        self.FORM_UP_THRESHOLD, self.FORM_DOWN_THRESHOLD = 45, 150
        self.SWAY_THRESHOLD_X, self.ELBOW_HIP_THRESHOLD_NORM = 25, 0.15
        self.SHOULDER_SHRUG_THRESHOLD, self.KNEE_ANGLE_THRESHOLD = 0.25, 165
 
    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="BicepCurls",
                      total_frames_analyzed=None,
                      analysis_timestamp=None,
                      metadata=None):
        if df.empty:
            # Always return full schema; safe defaults
            return {
                "total_reps_computed": 0,
                "feedback": {},
                "frame_info": {},
                "exercise_name": exercise_name,
                "analyzer_used": analyzer_used,
                "total_frames_analyzed": int(total_frames_analyzed or 0),
                "analysis_timestamp": analysis_timestamp,
                "metadata": _json_safe(metadata or {})
            }
        
        df = preprocess_data_for_analysis(df)
        
        baseline_hip_x = (df.iloc[0]['left_hip_x'] + df.iloc[0]['right_hip_x']) / 2
        current_rep_feedback = {'left': [], 'right': []}
        elbow_angle_at_peak = {'left': 180, 'right': 180}
 
        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            current_hip_x = (row['left_hip_x'] + row['right_hip_x']) / 2
            if abs(current_hip_x - baseline_hip_x) > self.SWAY_THRESHOLD_X:
                msg = "Body swaying. Brace your core to stay stable."
                current_rep_feedback['left'].append(msg)
                current_rep_feedback['right'].append(msg)
            
            for side in ['left', 'right']:
                if row[f'{side}_knee_angle'] < self.KNEE_ANGLE_THRESHOLD:
                    current_rep_feedback[side].append("Using leg momentum. Stand still and keep legs straight.")
                
                elbow_x, elbow_y = row[f'{side}_elbow_x'], row[f'{side}_elbow_y']
                hip_x, hip_y     = row[f'{side}_hip_x'],   row[f'{side}_hip_y']
                shoulder_y       = row[f'{side}_shoulder_y']
 
                torso_height = abs(hip_y - shoulder_y) if abs(hip_y - shoulder_y) > 0 else 1
                norm_dist = np.sqrt((elbow_x - hip_x)**2 + (elbow_y - hip_y)**2) / torso_height
                if norm_dist > self.ELBOW_HIP_THRESHOLD_NORM:
                    current_rep_feedback[side].append("Elbow moving forward. Keep it pinned to your side.")
 
                shoulder_width = abs(row['left_shoulder_x'] - row['right_shoulder_x'])
                if shoulder_width > 0:
                    shrug_ratio = abs(row[f'{side}_ear_y'] - row[f'{side}_shoulder_y']) / shoulder_width
                    if shrug_ratio < self.SHOULDER_SHRUG_THRESHOLD:
                        current_rep_feedback[side].append(f"{side.capitalize()} shoulder shrugging. Relax your traps.")
 
                elbow_angle = row[f'{side}_elbow_angle']
                
                # Transition to DOWN (rep completion)
                if elbow_angle > self.REP_COUNT_DOWN_THRESHOLD:
                    if self.stages[side] == 'up':
                        if elbow_angle_at_peak[side] > self.FORM_UP_THRESHOLD:
                            current_rep_feedback[side].append("Incomplete top contraction. Squeeze the bicep higher.")
                        feedback_to_add = current_rep_feedback[side] if current_rep_feedback[side] else ["Good Rep!"]
                        # preserve order, de-dup
                        self.feedback_log[side].append(_unique_preserve(feedback_to_add))
                        current_rep_feedback[side].clear()
                    self.stages[side] = "down"
 
                # Transition to UP (rep start)
                if elbow_angle < self.REP_COUNT_UP_THRESHOLD and self.stages[side] == 'down':
                    if local_idx > 0 and df.iloc[local_idx-1][f'{side}_elbow_angle'] < self.FORM_DOWN_THRESHOLD:
                        current_rep_feedback[side].append("Incomplete bottom extension. Straighten your arm fully.")
                    self.rep_counters[side] += 1
                    self.stages[side] = 'up'
                    self.frame_info[side][f"rep_{self.rep_counters[side]}"] = int(frame_num)
                    elbow_angle_at_peak[side] = elbow_angle
                
                if self.stages[side] == 'up':
                    elbow_angle_at_peak[side] = min(elbow_angle_at_peak[side], elbow_angle)
 
        # If the batch ends mid-rep, finalize the partial with current feedback
        for side in ['left', 'right']:
            if self.stages[side] == 'up' and len(self.feedback_log[side]) < self.rep_counters[side]:
                feedback_to_add = current_rep_feedback[side] if current_rep_feedback[side] else ["Good Rep!"]
                self.feedback_log[side].append(_unique_preserve(feedback_to_add))
 
        # Merge L/R into a canonical per-rep view
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
 
        # Choose side with more reps (tie -> left) and sort frame_info numerically
        final_side = 'left' if self.rep_counters['left'] >= self.rep_counters['right'] else 'right'
        final_frame_info = {
            k: int(v) for k, v in sorted(
                self.frame_info[final_side].items(),
                key=lambda kv: int(kv[0].split('_')[1])
            )
        }
 
        report = {
            "total_reps_computed": int(total_reps),
            "feedback": combined_feedback,
            "frame_info": final_frame_info,
            "exercise_name": exercise_name,
            "analyzer_used": analyzer_used,
            "total_frames_analyzed": int(total_frames_analyzed or len(df)),
            "analysis_timestamp": analysis_timestamp,
            "metadata": _json_safe(metadata or {})
        }
 
        # Final pass to ensure everything is JSON-safe (fix np.int64, etc.)
        report = _json_safe(report)
 
        # If you need a string for transport/logging:
        # json_str = json.dumps(report, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        # print(json_str)
 
        
        print(report)
        return report
 
# --- FEEDBACK AND TRIGGER CONDITIONS SUMMARY ---
#
# This summary explains every possible feedback message generated by this module and its actionable solution.
# The feedback is logged per-repetition in the final JSON report.
#
# POSITIVE FEEDBACK:
# ------------------
# "Good Rep!"
#   - Trigger: A full repetition is successfully completed (arm goes from 'down' to 'up' and back to 'down')
#     AND no other error feedback was logged during that specific repetition.
#
# ERROR FEEDBACK (BODY):
# ----------------------
# "Body swaying detected. Brace your core to stay stable."
#   - Trigger: The horizontal midpoint of the hips deviates more than SWAY_THRESHOLD_X (e.g., 25) pixels
#     from the baseline hip position established in the first frame of the batch.
#   - Why it's bad: Uses back momentum instead of isolating the biceps, risking injury.
#
# "Momentum from {side} leg. Stand still and keep legs straight."
#   - Trigger: The knee angle (formed by hip-knee-ankle) for the specified 'side' (left or right) becomes less than
#     KNEE_ANGLE_THRESHOLD (e.g., 165 degrees).
#   - Why it's bad: Uses leg drive to lift the weight, reducing the focus on the biceps.
#
# ERROR FEEDBACK (ARMS & SHOULDERS):
# -----------------------------------
# "Elbow moving forward. Keep it pinned to your side."
#   - Trigger: The normalized distance between the elbow and the hip exceeds ELBOW_HIP_THRESHOLD_NORM (e.g., 0.15).
#     The distance is normalized by the torso height to be robust to camera distance.
#   - Why it's bad: Engages the front deltoid (shoulder) muscle, taking the load off the bicep.
#
# "{Side} shoulder shrugging. Relax your traps."
#   - Trigger: The vertical distance between the ear and the shoulder becomes smaller than a ratio (e.g., 0.25) of the total shoulder width.
#   - Why it's bad: Uses the trapezius (neck/upper back) muscles to help lift, risking neck strain and reducing bicep work.
#
# "Incomplete top contraction. Squeeze the bicep higher."
#   - Trigger: During the 'up' phase of a repetition, the smallest elbow angle achieved is still greater than
#     UP_THRESHOLD + 15 (e.g., 45 + 15 = 60 degrees).
#   - Why it's bad: The bicep is not fully contracted, missing a key part of the muscle-building stimulus.
#
# "Incomplete bottom extension. Straighten your arm fully."
#   - Trigger: A new rep is started ('up' phase begins) but the elbow angle on the previous frame was not fully extended (i.e., was less than DOWN_THRESHOLD - 10).
#   - Why it's bad: Reduces the total range of motion and time under tension, making the exercise less effective.
#