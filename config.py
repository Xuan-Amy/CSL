from utils.io_utils import load_json
import joblib, numpy as np, cv2, mediapipe as mp

QUESTIONS_FILE = "data/questions.json"
USER_FILE = "data/users.json"

QUESTIONS = load_json(QUESTIONS_FILE)  # ✅ 确保初始化加载
camera_instances = {}

clf = joblib.load("models/svm_model.joblib")
scaler = joblib.load("models/scaler.joblib")
encoder = joblib.load("models/label_encoder.joblib")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def warmup_hands():
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = hands.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
