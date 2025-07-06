from flask import Flask, render_template, redirect, request, jsonify, Response
from flask_sock import Sock
from camera_ws import SignRecognizerWS
import json, os, joblib, numpy as np, cv2, mediapipe as mp, time

app = Flask(__name__)
sock = Sock(app)

# === 模型加载与 warmup ===
clf = joblib.load("models/svm_model.joblib")
scaler = joblib.load("models/scaler.joblib")
encoder = joblib.load("models/label_encoder.joblib")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def warmup_hands():
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = hands.process(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))

warmup_hands()

# === 数据加载 ===
QUESTIONS_FILE = "data/questions.json"
USER_FILE = "data/users.json"

def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

QUESTIONS = load_json(QUESTIONS_FILE)

# === 摄像头识别器实例池 ===
camera_instances = {}

# === 页面路由 ===
@app.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS)

@app.route("/task/<level>/<q>")
def task(level, q):
    question = QUESTIONS.get(level, {}).get(q)
    if not question:
        return f"题目 {level}-{q} 不存在", 404
    return render_template("task.html", level=level, q=q, prompt=question["prompt"], questions=QUESTIONS)

@app.route("/admin")
def admin():
    return render_template("admin.html", questions=QUESTIONS, users=load_json(USER_FILE))

@app.route("/admin/add_user", methods=["POST"])
def add_user():
    users = load_json(USER_FILE)
    name = request.form["username"]
    level = int(request.form.get("level", 1))
    question = int(request.form.get("question", 1))
    if name in users:
        return f"用户名 {name} 已存在", 400
    users[name] = {"unlocked_level": level, "unlocked_question": question}
    save_json(USER_FILE, users)
    return redirect("/admin")

@app.route("/admin/delete_user/<username>")
def delete_user(username):
    users = load_json(USER_FILE)
    users.pop(username, None)
    save_json(USER_FILE, users)
    return redirect("/admin")

@app.route("/admin/questions")
def admin_questions():
    return render_template("admin_questions.html", questions=QUESTIONS)

@app.route("/admin/add_question", methods=["POST"])
def add_question():
    level = request.form["level"]
    qid = request.form["qid"]
    prompt = request.form["prompt"]
    words = [w.strip() for w in request.form["words"].split(",") if w.strip()]
    motions = [m.strip() for m in request.form["motions"].split(",") if m.strip()]
    sequence = [{"word": w, "motion": motions[i] if i < len(motions) else None} for i, w in enumerate(words)]

    media_file = request.files.get("media")
    media_filename = None
    if media_file and media_file.filename:
        ext = os.path.splitext(media_file.filename)[1]
        media_filename = f"{level}-{qid}{ext}"
        save_path = os.path.join("static", "media", media_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media_file.save(save_path)

    if level not in QUESTIONS:
        QUESTIONS[level] = {}
    QUESTIONS[level][qid] = {"prompt": prompt, "target_sequence": sequence}
    if media_filename:
        QUESTIONS[level][qid]["media"] = media_filename
    save_json(QUESTIONS_FILE, QUESTIONS)
    return redirect("/admin/questions")

@app.route("/admin/delete_question/<level>/<qid>")
def delete_question(level, qid):
    if level in QUESTIONS and qid in QUESTIONS[level]:
        del QUESTIONS[level][qid]
        if not QUESTIONS[level]:
            del QUESTIONS[level]
        save_json(QUESTIONS_FILE, QUESTIONS)
    return redirect("/admin/questions")

@app.route("/admin/edit_question/<level>/<qid>")
def edit_question(level, qid):
    q = QUESTIONS.get(level, {}).get(qid)
    if not q:
        return "题目不存在", 404
    words = ",".join([i["word"] for i in q["target_sequence"]])
    motions = ",".join([i["motion"] or "" for i in q["target_sequence"]])
    return render_template("edit_question.html", level=level, qid=qid, question=q, words=words, motions=motions)

@app.route("/admin/edit_question/<level>/<qid>", methods=["POST"])
def update_question(level, qid):
    prompt = request.form["prompt"]
    words = [w.strip() for w in request.form["words"].split(",") if w.strip()]
    motions = [m.strip() for m in request.form["motions"].split(",") if m.strip()]
    sequence = [{"word": w, "motion": motions[i] if i < len(motions) else None} for i, w in enumerate(words)]

    if level not in QUESTIONS or qid not in QUESTIONS[level]:
        return "题目不存在", 404

    QUESTIONS[level][qid]["prompt"] = prompt
    QUESTIONS[level][qid]["target_sequence"] = sequence

    media_file = request.files.get("media")
    if media_file and media_file.filename:
        ext = os.path.splitext(media_file.filename)[1]
        media_filename = f"{level}-{qid}{ext}"
        save_path = os.path.join("static", "media", media_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media_file.save(save_path)
        QUESTIONS[level][qid]["media"] = media_filename

    save_json(QUESTIONS_FILE, QUESTIONS)
    return redirect("/admin/questions")

@app.route("/debug_ws")
def debug_ws():
    return render_template("debug_ws.html")

# ========== WebSocket 核心逻辑 ==========
def handle_ws(ws, recognizer):
    while True:
        try:
            data = ws.receive()
            if not data:
                break
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            result = recognizer.process_frame(frame)
            ws.send(json.dumps(result))
            time.sleep(0.01)
        except Exception as e:
            print(f"[ERROR] WebSocket 识别异常：{e}")
            ws.send(json.dumps({"error": f"识别错误：{str(e)}"}))
            break

# ========== WebSocket 接口 ==========
@sock.route("/ws/recognize")
def recognize_general(ws):
    recognizer = SignRecognizerWS([])
    handle_ws(ws, recognizer)

@sock.route("/ws/recognize/<level>/<qid>")
def recognize_task(ws, level, qid):
    key = f"{level}-{qid}"
    question = QUESTIONS.get(level, {}).get(qid)
    if not question:
        ws.send(json.dumps({"error": "题目不存在"}))
        return
    recognizer = SignRecognizerWS(question.get("target_sequence", []))
    camera_instances[key] = recognizer
    handle_ws(ws, recognizer)

@sock.route("/ws/recognize/debug/debug")
def recognize_debug(ws):
    recognizer = SignRecognizerWS([])
    camera_instances["debug-debug"] = recognizer
    handle_ws(ws, recognizer)

# ========== 状态轮询接口 ==========
@app.route("/status_feed/<level>/<qid>")
def status_feed(level, qid):
    key = f"{level}-{qid}"
    if key in camera_instances:
        status = camera_instances[key].get_status()
        return jsonify({
            "current": status.get("current", {}),
            "detected": status.get("detected", []),
            "completed": status.get("completed", False),
            "motion": {
                "left": camera_instances[key].motion.get_last_motion("left"),
                "right": camera_instances[key].motion.get_last_motion("right"),
            },
            "landmarks": {
                "left": camera_instances[key].last_landmarks.get("left", []),
                "right": camera_instances[key].last_landmarks.get("right", []),
            }
        })
    return jsonify({
        "current": {}, "detected": [], "completed": False,
        "motion": {"left": "-", "right": "-"},
        "landmarks": {"left": [], "right": []}
    })


# ========== 启动 ==========
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
