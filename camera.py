import cv2
import mediapipe as mp
import numpy as np
import joblib
from motiondetection import MotionDetector

class SignCamera:
    def __init__(self, target_sequence):
        """
        target_sequence: List[Dict]，如：
        [{"word": "车", "motion": "right"}, {"word": "房子", "motion": "up"}]
        """
        self.target_sequence = target_sequence
        self.detected = []

        self.clf = joblib.load("models/svm_model.joblib")
        self.scaler = joblib.load("models/scaler.joblib")
        self.encoder = joblib.load("models/label_encoder.joblib")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.motion = MotionDetector()
        self.trajectory = []  # 轨迹记录

        # 识别状态（供 status_feed 使用）
        self.status = {
            "current": "",
            "detected": [],
            "completed": False
        }

    def generate(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            label = ""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_num = len(results.multi_hand_landmarks)
                features = []

                for i in range(min(hand_num, 2)):
                    lm = results.multi_hand_landmarks[i].landmark
                    features.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])

                if hand_num == 1:
                    features.extend([-1] * (21 * 3))
                features.append(1 if hand_num == 2 else 0)

                # 更新移动轨迹
                wrist = results.multi_hand_landmarks[0].landmark[0]
                self.motion.update(wrist)

                h, w, _ = frame.shape
                self.trajectory.append((int(w * wrist.x), int(h * wrist.y)))
                if len(self.trajectory) > 30:
                    self.trajectory.pop(0)

                # 手语分类
                features_scaled = self.scaler.transform([features])
                pred = self.clf.predict(features_scaled)[0]
                label = self.encoder.inverse_transform([pred])[0]

                # 判断是否正确完成当前词+动作
                if len(self.detected) < len(self.target_sequence):
                    target_word = self.target_sequence[len(self.detected)]["word"]
                    target_motion = self.target_sequence[len(self.detected)]["motion"]

                    if label == target_word and self.motion.has_moved(target_motion):
                        self.detected.append(label)
                        self.motion.clear()
                        self.trajectory.clear()

            # 更新状态（给前端轮询）
            self.status["current"] = label
            self.status["detected"] = self.detected.copy()
            self.status["completed"] = len(self.detected) == len(self.target_sequence)

            # 仅绘制轨迹
            for i in range(1, len(self.trajectory)):
                cv2.line(frame, self.trajectory[i - 1], self.trajectory[i], (100, 255, 100), 4)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.cap.release()

    def get_status(self):
        return self.status
