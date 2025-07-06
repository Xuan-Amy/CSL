import joblib
import numpy as np
import cv2
import mediapipe as mp
from motiondetection import MotionDetector

class SignRecognizerWS:
    def __init__(self, target_sequence):
        self.target_sequence = target_sequence
        self.detected = []
        self.motion = MotionDetector()
        self.left_trajectory = []
        self.right_trajectory = []

        # 加载模型
        self.clf = joblib.load("models/svm_model.joblib")
        self.scaler = joblib.load("models/scaler.joblib")
        self.encoder = joblib.load("models/label_encoder.joblib")

        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def sanitize_landmarks(self, landmarks):
        """确保 landmarks 数据可序列化，无 NaN 等异常值"""
        safe = []
        for pt in landmarks:
            try:
                x, y = float(pt["x"]), float(pt["y"])
                if np.isnan(x) or np.isnan(y):
                    continue
                safe.append({"x": x, "y": y})
            except:
                continue
        return safe

    def process_frame(self, frame):
        try:
            label_left, label_right = "", ""
            landmarks_dict = {"left": [], "right": []}
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            h, w = frame.shape[:2]

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_map = {}
                for i, h_info in enumerate(results.multi_handedness):
                    hand_label = h_info.classification[0].label  # "Left" or "Right"
                    hand_lms = results.multi_hand_landmarks[i]
                    if not hand_lms or not hand_lms.landmark or len(hand_lms.landmark) != 21:
                        continue
                    hand_map[hand_label] = hand_lms

                # 特征提取
                features = []
                for label in ["Left", "Right"]:
                    if label in hand_map:
                        lm = hand_map[label].landmark
                        features.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])
                    else:
                        features.extend([-1] * 63)
                features.append(1 if len(hand_map) == 2 else 0)

                # 模型推理
                features_scaled = self.scaler.transform([features])
                pred = self.clf.predict(features_scaled)[0]
                label_full = self.encoder.inverse_transform([pred])[0]

                # 手势轨迹与动作判断
                for hand_label, landmarks in hand_map.items():
                    hand_key = "left" if hand_label == "Left" else "right"
                    lm_points = []
                    for pt in landmarks.landmark:
                        x_px, y_px = int(pt.x * w), int(pt.y * h)
                        lm_points.append({"x": x_px, "y": y_px})
                    landmarks_dict[hand_key] = self.sanitize_landmarks(lm_points)

                    # wrist (0)
                    wrist = landmarks.landmark[0]
                    point = (int(wrist.x * w), int(wrist.y * h))

                    if hand_label == "Left":
                        self.left_trajectory.append(point)
                        if len(self.left_trajectory) > 30:
                            self.left_trajectory.pop(0)
                        self.motion.update(left_hand=(landmarks.landmark[0], landmarks.landmark[8], landmarks.landmark[4]))
                        label_left = label_full
                    else:
                        self.right_trajectory.append(point)
                        if len(self.right_trajectory) > 30:
                            self.right_trajectory.pop(0)
                        self.motion.update(right_hand=(landmarks.landmark[0], landmarks.landmark[8], landmarks.landmark[4]))
                        label_right = label_full

                # 匹配任务序列
                if self.target_sequence and len(self.detected) < len(self.target_sequence):
                    target_word = self.target_sequence[len(self.detected)]["word"]
                    target_motion = self.target_sequence[len(self.detected)]["motion"]
                    for hand in ["left", "right"]:
                        current_label = label_left if hand == "left" else label_right
                        if current_label == target_word and self.motion.has_moved(hand, target_motion):
                            self.detected.append(current_label)
                            self.motion.clear()
                            self.left_trajectory.clear()
                            self.right_trajectory.clear()
                            break
            else:
                self.left_trajectory.clear()
                self.right_trajectory.clear()
                self.motion.clear()

            current_motion = ""
            if self.target_sequence and len(self.detected) < len(self.target_sequence):
                current_motion = self.target_sequence[len(self.detected)]["motion"]

            return {
                "left": label_left,
                "right": label_right,
                "landmarks": {
                    "left": self.sanitize_landmarks(landmarks_dict["left"]),
                    "right": self.sanitize_landmarks(landmarks_dict["right"])
                },
                "status": {
                    "current": {
                        "left": label_left,
                        "right": label_right
                    },
                    "motion": {
                        "left": self.motion.get_last_motion("left"),
                        "right": self.motion.get_last_motion("right")
                    },
                    "detected": self.detected.copy(),
                    "completed": len(self.detected) == len(self.target_sequence)
                }
            }

        except Exception as e:
            print(f"[⚠️ process_frame 异常] {e}")
            return {
                "left": "",
                "right": "",
                "landmarks": {"left": [], "right": []},
                "status": {
                    "current": {"left": "-", "right": "-"},
                    "motion": {"left": "-", "right": "-"},
                    "detected": self.detected.copy(),
                    "completed": False,
                    "error": str(e)
                }
            }
