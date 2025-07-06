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

    def process_frame(self, frame):
        label_left, label_right = "", ""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_map = {}
            for i, h_info in enumerate(results.multi_handedness):
                hand_label = h_info.classification[0].label
                hand_map[hand_label] = results.multi_hand_landmarks[i]

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
                wrist = landmarks.landmark[0]
                point = (int(wrist.x * 640), int(wrist.y * 480))  # 假设图像大小

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

        # 当前目标动作（optional 可用）
        current_motion = ""
        if self.target_sequence and len(self.detected) < len(self.target_sequence):
            current_motion = self.target_sequence[len(self.detected)]["motion"]

        return {
            "left": label_left,
            "right": label_right,
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
