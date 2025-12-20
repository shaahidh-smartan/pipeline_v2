# modules/kb_goblet_squats.py

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

class KB_Goblet_Squats:
    def __init__(self):
        self.stage = "up"
        self.counter = 0
        self.feedback_log = []
        self.frame_info = {}

        self.REP_COUNT_UP_THRESHOLD, self.REP_COUNT_DOWN_THRESHOLD = 150, 110
        self.FORM_UP_THRESHOLD = 165
        self.TORSO_UPRIGHT_THRESHOLD = 65
        self.HEEL_LIFT_THRESHOLD = 15

    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="KB_Goblet_Squats",
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
        is_deep_enough_in_rep = False
        baseline_ankle_y = (df.iloc[0]['left_ankle_y'] + df.iloc[0]['right_ankle_y']) / 2

        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            
            avg_knee_angle = (row['left_knee_angle'] + row['right_knee_angle']) / 2
            avg_torso_angle = (row['left_hip_angle'] + row['right_hip_angle']) / 2
            
            current_ankle_y = (row['left_ankle_y'] + row['right_ankle_y']) / 2
            if baseline_ankle_y - current_ankle_y > self.HEEL_LIFT_THRESHOLD:
                current_rep_feedback.append("Heels are lifting. Sit back and keep your feet flat.")

            if avg_knee_angle > self.REP_COUNT_UP_THRESHOLD:
                if self.stage == 'down':
                    if not is_deep_enough_in_rep:
                        current_rep_feedback.append("Squat deeper. Your hips should go below your knees.")
                    if row['left_hip_angle'] < self.FORM_UP_THRESHOLD or row['right_hip_angle'] < self.FORM_UP_THRESHOLD:
                        current_rep_feedback.append("Incomplete hip extension. Stand up fully.")
                    feedback_to_add = current_rep_feedback if current_rep_feedback else ["Good Rep!"]
                    self.feedback_log.append(_unique_preserve(feedback_to_add))
                    current_rep_feedback.clear()
                self.stage = "up"
            
            if avg_knee_angle < self.REP_COUNT_DOWN_THRESHOLD and self.stage == 'up':
                self.stage = 'down'
                self.counter += 1
                self.frame_info[f"rep_{self.counter}"] = int(frame_num)
                is_deep_enough_in_rep = False

            if self.stage == 'down':
                if row['left_hip_y'] > row['left_knee_y'] and row['right_hip_y'] > row['right_knee_y']:
                    is_deep_enough_in_rep = True

                knee_distance = abs(row['left_knee_x'] - row['right_knee_x'])
                ankle_distance = abs(row['left_ankle_x'] - row['right_ankle_x'])
                if ankle_distance > 0 and knee_distance < ankle_distance * 0.9:
                    current_rep_feedback.append("Knees collapsing inward. Push them out.")

                if avg_torso_angle < self.TORSO_UPRIGHT_THRESHOLD:
                    current_rep_feedback.append("Chest falling forward. Keep your torso upright.")

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
# "Chest falling forward. Keep your torso upright."
#   - Trigger: During the 'down' phase, the average torso angle (shoulder-hip-knee) is less than TORSO_UPRIGHT_THRESHOLD.
#   - Why it's bad: Places excessive strain on the lower back and reduces the effectiveness for the legs and glutes.
#
# "Incomplete hip extension. Stand up fully."
#   - Trigger: A rep is completed, but the hip angle at the top was less than the strict FORM_UP_THRESHOLD.
#   - Why it's bad: Fails to fully engage the glutes at the top of the movement.
#
# ERROR FEEDBACK (LEGS):
# "Squat deeper. Your hips should go below your knees."
#   - Trigger: A rep is completed, but the hip's y-coordinate never went below the knee's y-coordinate during the 'down' phase.
#   - Why it's bad: This is a "partial rep" that fails to achieve a full range of motion, reducing muscle activation.
#
# "Knees collapsing inward. Push them out."
#   - Trigger: During the 'down' phase, the horizontal distance between the knees becomes significantly less than the distance between the ankles.
#   - Why it's bad: This is known as knee valgus and places dangerous stress on the knee ligaments (ACL/MCL).
#
# "Heels are lifting. Sit back and keep your feet flat."
#   - Trigger: The average vertical position of the ankles moves up more than HEEL_LIFT_THRESHOLD from their starting position.
#   - Why it's bad: Indicates a loss of balance and stability, shifting stress from the glutes and hamstrings to the quads and knees.
#