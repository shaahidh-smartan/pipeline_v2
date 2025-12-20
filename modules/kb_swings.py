# modules/kb_swings.py

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

class KB_Swings:
    def __init__(self):
        self.stage = "up"
        self.counter = 0
        self.feedback_log = []
        self.frame_info = {}

        self.REP_COUNT_UP_THRESHOLD, self.REP_COUNT_DOWN_THRESHOLD = 150, 100
        self.FORM_UP_THRESHOLD, self.FORM_DOWN_THRESHOLD = 160, 90
        self.KNEE_BEND_THRESHOLD = 135
        self.ARM_LIFT_THRESHOLD = 45

    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="KB_Swings",
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
            
        current_rep_feedback = []
        hip_angle_at_bottom = 180

        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            
            side = 'left' if row['left_hip_x'] > 0 and row['left_knee_x'] > 0 else 'right'
            if not (row[f'{side}_hip_x'] > 0 and row[f'{side}_knee_x'] > 0): continue

            hip_angle = row[f'{side}_hip_angle']
            knee_angle = row[f'{side}_knee_angle']
            arm_torso_angle = row[f'{side}_raise_angle']
            
            if hip_angle > self.REP_COUNT_UP_THRESHOLD:
                if self.stage == 'down':
                    if hip_angle_at_bottom > self.FORM_DOWN_THRESHOLD:
                        current_rep_feedback.append("Incomplete hinge. Push your hips back further.")
                    feedback_to_add = current_rep_feedback if current_rep_feedback else ["Good Rep!"]
                    self.feedback_log.append(_unique_preserve(feedback_to_add))
                    current_rep_feedback.clear()
                self.stage = "up"
            
            if hip_angle < self.REP_COUNT_DOWN_THRESHOLD and self.stage == 'up':
                if local_idx > 0 and df.iloc[local_idx-1][f'{side}_hip_angle'] < self.FORM_UP_THRESHOLD:
                    current_rep_feedback.append("Incomplete hip extension. Squeeze your glutes at the top.")
                self.stage = 'down'
                self.counter += 1
                self.frame_info[f"rep_{self.counter}"] = int(frame_num)
                hip_angle_at_bottom = hip_angle

            if self.stage == 'down':
                hip_angle_at_bottom = min(hip_angle_at_bottom, hip_angle)
                if knee_angle < self.KNEE_BEND_THRESHOLD:
                    current_rep_feedback.append("Squatting the weight. Hinge at your hips instead of bending your knees.")
            
            if arm_torso_angle > self.ARM_LIFT_THRESHOLD:
                current_rep_feedback.append("Lifting with arms. Let your hips generate the power.")
            wrist_y, shoulder_y = row[f'{side}_wrist_y'], row[f'{side}_shoulder_y']
            if wrist_y < shoulder_y:
                current_rep_feedback.append("Swinging too high. Aim for chest level.")

        if self.stage == 'down' and len(self.feedback_log) < self.counter:
            self.feedback_log.append(_unique_preserve(current_rep_feedback if current_rep_feedback else ["Good Rep!"]))

        combined_feedback = {}
        for i in range(1, self.counter + 1):
            feedback_list = self.feedback_log[i-1] if i-1 < len(self.feedback_log) else []
            if len(feedback_list) > 1 and "Good Rep!" in feedback_list:
                feedback_list.remove("Good Rep!")
            combined_feedback[f"rep_{i}"] = feedback_list if feedback_list else ["Good Rep!"]

        final_frame_info = {
            k: v for k, v in sorted(
                self.frame_info.items(),
                key=lambda kv: int(kv[0].split('_')[1])
            )
        }

        report = { 
            "total_reps_computed": self.counter, "feedback": combined_feedback, "frame_info": final_frame_info,
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
# "Squatting the weight. Hinge at your hips instead of bending your knees."
#   - Trigger: During the 'down' phase, the knee angle bends more than the KNEE_BEND_THRESHOLD.
#   - Why it's bad: This is the primary error. It turns a posterior chain (glutes/hamstrings) exercise into a leg exercise, defeating the purpose.
#
# "Incomplete hinge. Push your hips back further."
#   - Trigger: A rep is completed, but the lowest hip angle achieved was greater than the strict FORM_DOWN_THRESHOLD.
#   - Why it's bad: Fails to properly load the hamstrings and glutes for an explosive contraction.
#
# "Incomplete hip extension. Squeeze your glutes at the top."
#   - Trigger: A new rep is started, but the hip angle in the previous frame was less than the strict FORM_UP_THRESHOLD.
#   - Why it's bad: Fails to achieve a powerful lockout, which is the goal of the movement.
#
# ERROR FEEDBACK (ARMS & SHOULDERS):
# "Lifting with arms. Let your hips generate the power."
#   - Trigger: The angle between the torso and the arm (hip-shoulder-wrist) exceeds the ARM_LIFT_THRESHOLD.
#   - Why it's bad: Indicates the user is pulling the weight up with their shoulders instead of using it as a pendulum driven by the hips.
#
# "Swinging too high. Aim for chest level."
#   - Trigger: The wrist's y-coordinate becomes higher than the shoulder's y-coordinate.
#   - Why it's bad: Can indicate a lack of control and may place unnecessary stress on the lower back at the peak.
#