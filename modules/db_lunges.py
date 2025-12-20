# modules/db_lunges.py

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

class DB_Lunges:
    def __init__(self):
        self.stages = {'left': 'up', 'right': 'up'}
        self.rep_counters = {'left': 0, 'right': 0}
        self.feedback_log = {'left': [], 'right': []}
        self.frame_info = {'left': {}, 'right': {}}

        self.REP_COUNT_DOWN_THRESHOLD, self.REP_COUNT_UP_THRESHOLD = 120, 150
        self.FORM_DOWN_FRONT_KNEE, self.FORM_DOWN_BACK_KNEE = 100, 100
        self.FORM_UP_THRESHOLD = 160
        self.TORSO_LEAN_THRESHOLD = 25
        self.KNEE_OVER_ANKLE_TOLERANCE_X = 30

    def analyze_batch(self, df, *,
                      exercise_name=None,
                      analyzer_used="DB_Lunges",
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
        knee_angle_at_bottom = {'left': 180, 'right': 180}

        for local_idx, (_, row) in enumerate(df.iterrows()):
            frame_num = row['frame_rel_idx']
            
            # Determine which leg is forward based on hip position (assuming side view)
            front_leg = 'left' if row['left_hip_x'] > row['right_hip_x'] else 'right'
            back_leg = 'right' if front_leg == 'left' else 'left'

            front_knee_angle = row[f'{front_leg}_knee_angle']
            back_knee_angle = row[f'{back_leg}_knee_angle']
            
            if back_knee_angle > self.REP_COUNT_UP_THRESHOLD and front_knee_angle > self.REP_COUNT_UP_THRESHOLD:
                if self.stages[front_leg] == 'down':
                    front_knee_at_bottom, back_knee_at_bottom = knee_angle_at_bottom[front_leg], knee_angle_at_bottom[back_leg]
                    if not (80 < front_knee_at_bottom < self.FORM_DOWN_FRONT_KNEE):
                        current_rep_feedback[front_leg].append("Lunge deeper with your front leg.")
                    if not (80 < back_knee_at_bottom < self.FORM_DOWN_BACK_KNEE):
                        current_rep_feedback[front_leg].append("Lower your back knee closer to the ground.")
                    
                    if local_idx > 0 and (df.iloc[local_idx-1]['left_hip_angle'] < self.FORM_UP_THRESHOLD or df.iloc[local_idx-1]['right_hip_angle'] < self.FORM_UP_THRESHOLD):
                        current_rep_feedback[front_leg].append("Incomplete hip extension. Stand up fully.")
                    
                    feedback_to_add = current_rep_feedback[front_leg] if current_rep_feedback[front_leg] else ["Good Rep!"]
                    self.feedback_log[front_leg].append(_unique_preserve(feedback_to_add))
                    current_rep_feedback[front_leg].clear()
                self.stages['left'], self.stages['right'] = 'up', 'up'

            if front_knee_angle < self.REP_COUNT_DOWN_THRESHOLD and back_knee_angle < self.REP_COUNT_DOWN_THRESHOLD and self.stages[front_leg] == 'up':
                self.stages[front_leg] = 'down'
                self.rep_counters[front_leg] += 1
                self.frame_info[front_leg][f"rep_{self.rep_counters[front_leg]}"] = int(frame_num)
                knee_angle_at_bottom[front_leg], knee_angle_at_bottom[back_leg] = front_knee_angle, back_knee_angle

            if self.stages[front_leg] == 'down':
                knee_angle_at_bottom[front_leg] = min(knee_angle_at_bottom[front_leg], front_knee_angle)
                knee_angle_at_bottom[back_leg] = min(knee_angle_at_bottom[back_leg], back_knee_angle)
                
                if row[f'{front_leg}_knee_x'] > row[f'{front_leg}_ankle_x'] + self.KNEE_OVER_ANKLE_TOLERANCE_X:
                    current_rep_feedback[front_leg].append(f"Don't let your front knee go past your toes.")
                
                shoulder_pt, hip_pt = (row[f'{front_leg}_shoulder_x'], row[f'{front_leg}_shoulder_y']), (row[f'{front_leg}_hip_x'], row[f'{front_leg}_hip_y'])
                vertical_pt = (hip_pt[0], hip_pt[1] - 100)
                torso_vertical_angle = calculate_angle(shoulder_pt, hip_pt, vertical_pt)
                if torso_vertical_angle > self.TORSO_LEAN_THRESHOLD:
                    current_rep_feedback[front_leg].append("Keep your torso upright.")

        for side in ['left', 'right']:
            if self.stages[side] == 'down' and len(self.feedback_log[side]) < self.rep_counters[side]:
                self.feedback_log[side].append(_unique_preserve(current_rep_feedback[side] if current_rep_feedback[side] else ["Good Rep!"]))

        # --- Format the final JSON output, sorted chronologically ---
        total_reps = self.rep_counters['left'] + self.rep_counters['right']
        all_reps = []
        for side in ['left', 'right']:
            for rep, frame in self.frame_info[side].items():
                rep_num_side = int(rep.split('_')[1])
                all_reps.append({
                    'side': side,
                    'rep_num_side': rep_num_side,
                    'frame': frame,
                    'feedback': self.feedback_log[side][rep_num_side - 1]
                })
        
        sorted_reps = sorted(all_reps, key=lambda x: x['frame'])
        combined_feedback, final_frame_info = {}, {}
        for i, rep_data in enumerate(sorted_reps):
            rep_key = f"rep_{i+1}"
            feedback_list = rep_data['feedback']
            if len(feedback_list) > 1 and "Good Rep!" in feedback_list: feedback_list.remove("Good Rep!")
            
            # Add the side as the first piece of feedback
            final_feedback = [f"({rep_data['side'].capitalize()})"] + feedback_list if feedback_list else [f"({rep_data['side'].capitalize()}) Good Rep!"]
            combined_feedback[rep_key] = final_feedback
            final_frame_info[rep_key] = rep_data['frame']

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
#   - Trigger: A full repetition is successfully completed on one leg with no other error feedback logged for that rep.
#
# ERROR FEEDBACK (BODY):
# "Keep your torso upright."
#   - Trigger: The angle of the torso (shoulder-hip line) deviates more than TORSO_LEAN_THRESHOLD from a pure vertical line.
#   - Why it's bad: Leaning too far forward places excessive strain on the lower back.
#
# "Incomplete hip extension. Stand up fully."
#   - Trigger: A rep is completed, but the hip angle at the top was less than the strict FORM_UP_THRESHOLD.
#   - Why it's bad: Fails to fully engage the glute of the front leg, which is a primary target of the exercise.
#
# ERROR FEEDBACK (LEGS):
# "Lunge deeper with your front leg."
#   - Trigger: A rep is completed, but the lowest front knee angle achieved was greater than the strict FORM_DOWN_FRONT_KNEE threshold.
#   - Why it's bad: This is a partial rep that doesn't fully engage the quadriceps and glutes.
#
# "Lower your back knee closer to the ground."
#   - Trigger: A rep is completed, but the lowest back knee angle achieved was greater than the strict FORM_DOWN_BACK_KNEE threshold.
#   - Why it's bad: This indicates insufficient depth for a full range of motion.
#
# "Don't let your front knee go past your toes."
#   - Trigger: During the 'down' phase, the front knee's x-coordinate moves significantly past the front ankle's x-coordinate.
#   - Why it's bad: This is a major safety risk that places excessive shear force on the knee joint.
#