# modules/singlearm_db_rows.py

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
        'left_shoulder_angle': ('left_hip', 'left_shoulder', 'left_elbow'),
        'right_shoulder_angle': ('right_hip', 'right_shoulder', 'right_elbow'),
        'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
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

class Singlearm_DB_Rows:
    def __init__(self):
        self.stage = "down"
        self.counter = 0
        self.feedback_log = []
        self.frame_info = {}

        self.REP_COUNT_UP_THRESHOLD, self.REP_COUNT_DOWN_THRESHOLD = 50, 30
        self.FORM_UP_THRESHOLD, self.FORM_DOWN_THRESHOLD = 60, 25
        self.BACK_FLAT_MIN_ANGLE, self.BACK_FLAT_MAX_ANGLE = 70, 110
        self.YANK_THRESHOLD = 20
        self.ARM_EXTENSION_THRESHOLD = 150

    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="Singlearm_DB_Rows",
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
        row_angle_at_peak = 0
        
        # Determine the baseline shoulder y from the first visible side in the first frame
        first_row = df.iloc[0]
        baseline_shoulder_y = first_row['left_shoulder_y'] if first_row['left_hip_x'] > 0 else first_row['right_shoulder_y']

        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            
            side = 'left' if row['left_shoulder_x'] > 0 and row['left_hip_x'] > 0 else 'right'
            if not (row[f'{side}_shoulder_x'] > 0 and row[f'{side}_hip_x'] > 0): continue

            back_angle, row_angle, elbow_angle, shoulder_y = row[f'{side}_hip_angle'], row[f'{side}_shoulder_angle'], row[f'{side}_elbow_angle'], row[f'{side}_shoulder_y']

            if not self.BACK_FLAT_MIN_ANGLE < back_angle < self.BACK_FLAT_MAX_ANGLE:
                current_rep_feedback.append("Back angle is changing. Keep your torso stable and flat.")
            if abs(shoulder_y - baseline_shoulder_y) > self.YANK_THRESHOLD:
                current_rep_feedback.append("Yanking the weight. Use your back, not momentum.")

            if row_angle > self.REP_COUNT_UP_THRESHOLD and self.stage == 'down':
                if local_idx > 0 and df.iloc[local_idx-1][f'{side}_elbow_angle'] < self.ARM_EXTENSION_THRESHOLD:
                    current_rep_feedback.append("Incomplete bottom stretch. Let your arm hang fully.")
                self.stage = 'up'
                self.counter += 1
                self.frame_info[f"rep_{self.counter}"] = int(frame_num)

            if row_angle < self.REP_COUNT_DOWN_THRESHOLD and self.stage == 'up':
                if row_angle_at_peak < self.FORM_UP_THRESHOLD:
                    current_rep_feedback.append("Incomplete contraction. Pull your elbow higher and back.")
                feedback_to_add = current_rep_feedback if current_rep_feedback else ["Good Rep!"]
                self.feedback_log.append(_unique_preserve(feedback_to_add))
                current_rep_feedback.clear()
                self.stage = "down"
                row_angle_at_peak = 0
            
            if self.stage == 'up':
                row_angle_at_peak = max(row_angle_at_peak, row_angle)

        if self.stage == 'up' and len(self.feedback_log) < self.counter:
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
# "Back angle is changing. Keep your torso stable and flat."
#   - Trigger: The hip hinge angle (shoulder-hip-knee) goes outside the stable BENT_OVER range.
#   - Why it's bad: Indicates the user is standing up or rounding their back during the rep, which uses momentum and risks injury.
#
# "Yanking the weight. Use your back, not momentum."
#   - Trigger: The vertical position of the shoulder moves more than the YANK_THRESHOLD.
#   - Why it's bad: Shows a lack of control and relies on momentum instead of muscle contraction.
#
# ERROR FEEDBACK (ARM & SHOULDER):
# "Incomplete bottom stretch. Let your arm hang fully."
#   - Trigger: A new rep is started, but the elbow angle on the previous frame was not fully extended (straighter than ARM_EXTENSION_THRESHOLD).
#   - Why it's bad: This is a "partial rep" that doesn't fully stretch the latissimus dorsi muscle, reducing the overall effectiveness.
#
# "Incomplete contraction. Pull your elbow higher and back."
#   - Trigger: A rep is completed, but the peak row angle achieved (hip-shoulder-elbow) was less than the strict FORM_UP_THRESHOLD.
#   - Why it's bad: Fails to fully contract the back muscles at the peak of the movement.
#