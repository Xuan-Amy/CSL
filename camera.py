import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import time
from motiondetection import MotionDetector
from PIL import ImageFont, ImageDraw, Image

def draw_chinese_text(frame, text, position, font_path="simhei.ttf", font_size=22, color=(255, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def compute_curvature(trajectory):
    if len(trajectory) < 3:
        return 0.0
    traj = np.array(trajectory, dtype=np.float32)
    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
    return float(np.mean(curvature))

class SignCamera:
    def __init__(self, target_sequence):
        self.target_sequence = target_sequence
        self.detected = []
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()

        self.status = {
            "current": "",
            "detected": [],
            "completed": False
        }

        # 初始化模型
        self.clf = joblib.load("models/svm_model.joblib")
        self.scaler = joblib.load("models/scaler.joblib")
        self.encoder = joblib.load("models/label_encoder.joblib")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        self.motion = MotionDetector()
        self.left_trajectory = []
        self.right_trajectory = []

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            processed = self._process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed)
            if not ret:
                continue

            with self.lock:
                self.latest_frame = buffer.tobytes()

            time.sleep(0.03)  # 控制帧率，降低 CPU 占用

    def _process_frame(self, frame):
        label_left, label_right = "", ""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = self.hands.process(frame_rgb)
        except Exception as e:
            print("Mediapipe Error:", e)
            return frame

        h, w, _ = frame.shape
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks
            handedness = results.multi_handedness

            hand_map = {}
            for i, h_info in enumerate(handedness):
                hand_label = h_info.classification[0].label
                hand_map[hand_label] = hand_landmarks[i]

            features = []
            for label in ["Left", "Right"]:
                if label in hand_map:
                    lm = hand_map[label].landmark
                    features.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])
                else:
                    features.extend([-1] * (21 * 3))
            features.append(1 if len(hand_map) == 2 else 0)

            features_scaled = self.scaler.transform([features])
            pred = self.clf.predict(features_scaled)[0]
            label_full = self.encoder.inverse_transform([pred])[0]

            for hand_label, landmarks in hand_map.items():
                lm = landmarks.landmark
                color = (255, 0, 0) if hand_label == "Left" else (0, 255, 0)
                pts = [(int(pt.x * w), int(pt.y * h)) for pt in lm]
                xs, ys = zip(*pts)
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                wrist = lm[0]
                point = (int(wrist.x * w), int(wrist.y * h))

                if hand_label == "Left":
                    self.left_trajectory.append(point)
                    if len(self.left_trajectory) > 30:
                        self.left_trajectory.pop(0)
                    self.motion.update(left_hand=(lm[0], lm[8], lm[4]))
                    label_left = label_full
                else:
                    self.right_trajectory.append(point)
                    if len(self.right_trajectory) > 30:
                        self.right_trajectory.pop(0)
                    self.motion.update(right_hand=(lm[0], lm[8], lm[4]))
                    label_right = label_full

                traj = self.left_trajectory if hand_label == "Left" else self.right_trajectory
                curvature = compute_curvature(traj)
                text = f"{hand_label}: {label_full} (K={curvature:.2f})"
                frame = draw_chinese_text(frame, text, (xmin, ymin - 30), font_path="simhei.ttf", font_size=22, color=color)

            # Debug 模式
            if not self.target_sequence:
                for label in [label_left, label_right]:
                    if label and label not in self.detected:
                        self.detected.append(label)
            # 正常识别序列
            elif len(self.detected) < len(self.target_sequence):
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
            label_left, label_right = "", ""
            self.left_trajectory.clear()
            self.right_trajectory.clear()
            self.motion.clear()

        # 轨迹绘制
        def draw_fading_trail(trajectory, base_color):
            trail_len = len(trajectory)
            for i in range(1, trail_len):
                alpha = i / trail_len
                color = tuple(int(c * alpha + 255 * (1 - alpha)) for c in base_color)
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

        draw_fading_trail(self.left_trajectory, (255, 0, 0))
        draw_fading_trail(self.right_trajectory, (0, 255, 0))

        self.status["current"] = {"left": label_left, "right": label_right}
        self.status["detected"] = self.detected.copy()
        self.status["completed"] = len(self.detected) == len(self.target_sequence)

        return frame

    def generate(self):
        """ 用于 MJPEG 流输出 """
        while self.running:
            with self.lock:
                frame = self.latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)

    def get_status(self):
        return self.status

    def __del__(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
