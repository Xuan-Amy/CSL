from flask import Flask, render_template, redirect, url_for, Response, request, jsonify
import json
import os
from camera import SignCamera

app = Flask(__name__)

# === 加载题库 ===
QUESTIONS_FILE = "data/questions.json"
with open(QUESTIONS_FILE, encoding="utf-8") as f:
    QUESTIONS = json.load(f)

def save_questions(data):
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# === 用户数据 ===
USER_FILE = "data/users.json"
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(data):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# === 摄像头识别实例池 ===
camera_instances = {}

# === 前台页面 ===
@app.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS)

@app.route("/task/<level>/<q>")
def task(level, q):
    question = QUESTIONS.get(level, {}).get(q)
    if not question:
        return f"题目 {level}-{q} 不存在", 404
    return render_template("task.html", level=level, q=q, prompt=question["prompt"], questions=QUESTIONS)


@app.route("/video_feed/<level>/<q>")
def video_feed(level, q):
    key = f"{level}-{q}"
    if key not in camera_instances:
        target_sequence = QUESTIONS[level][q]["target_sequence"]
        camera_instances[key] = SignCamera(target_sequence)
    return Response(camera_instances[key].generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status_feed/<level>/<q>")
def status_feed(level, q):
    key = f"{level}-{q}"
    if key in camera_instances:
        return jsonify(camera_instances[key].get_status())
    return jsonify({"current": "", "detected": [], "completed": False})

# === 管理员首页 ===
@app.route("/admin")
def admin():
    users = load_users()
    return render_template("admin.html", questions=QUESTIONS, users=users)

# === 用户管理 ===
@app.route("/admin/add_user", methods=["POST"])
def add_user():
    username = request.form.get("username")
    level = int(request.form.get("level", 1))
    question = int(request.form.get("question", 1))

    users = load_users()
    if username in users:
        return f"⚠️ 用户名 {username} 已存在", 400

    users[username] = {
        "unlocked_level": level,
        "unlocked_question": question
    }
    save_users(users)
    return redirect("/admin")

@app.route("/admin/delete_user/<username>")
def delete_user(username):
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
    return redirect("/admin")

# === 题库管理 ===
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

    sequence = []
    for i, word in enumerate(words):
        motion = motions[i] if i < len(motions) and motions[i] else None
        sequence.append({"word": word, "motion": motion})

    media_file = request.files.get("media")
    media_filename = None
    if media_file and media_file.filename != "":
        ext = os.path.splitext(media_file.filename)[1]
        media_filename = f"{level}-{qid}{ext}"
        save_path = os.path.join("static", "media", media_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media_file.save(save_path)

    if level not in QUESTIONS:
        QUESTIONS[level] = {}
    QUESTIONS[level][qid] = {
        "prompt": prompt,
        "target_sequence": sequence
    }
    if media_filename:
        QUESTIONS[level][qid]["media"] = media_filename

    save_questions(QUESTIONS)
    return redirect("/admin/questions")

@app.route("/admin/delete_question/<level>/<qid>")
def delete_question(level, qid):
    if level in QUESTIONS and qid in QUESTIONS[level]:
        del QUESTIONS[level][qid]
        if not QUESTIONS[level]:
            del QUESTIONS[level]
        save_questions(QUESTIONS)
    return redirect("/admin/questions")

@app.route("/admin/edit_question/<level>/<qid>")
def edit_question(level, qid):
    question = QUESTIONS.get(level, {}).get(qid)
    if not question:
        return "题目不存在", 404
    words = ",".join([item["word"] for item in question["target_sequence"]])
    motions = ",".join([item["motion"] or "" for item in question["target_sequence"]])
    return render_template("edit_question.html", level=level, qid=qid, question=question, words=words, motions=motions)

@app.route("/admin/edit_question/<level>/<qid>", methods=["POST"])
def update_question(level, qid):
    prompt = request.form["prompt"]
    words = [w.strip() for w in request.form["words"].split(",") if w.strip()]
    motions = [m.strip() for m in request.form["motions"].split(",") if m.strip()]

    sequence = []
    for i, word in enumerate(words):
        motion = motions[i] if i < len(motions) and motions[i] else None
        sequence.append({"word": word, "motion": motion})

    if level not in QUESTIONS:
        QUESTIONS[level] = {}
    if qid not in QUESTIONS[level]:
        return "题目不存在", 404

    QUESTIONS[level][qid]["prompt"] = prompt
    QUESTIONS[level][qid]["target_sequence"] = sequence

    media_file = request.files.get("media")
    if media_file and media_file.filename != "":
        ext = os.path.splitext(media_file.filename)[1]
        media_filename = f"{level}-{qid}{ext}"
        save_path = os.path.join("static", "media", media_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media_file.save(save_path)
        QUESTIONS[level][qid]["media"] = media_filename

    save_questions(QUESTIONS)
    return redirect("/admin/questions")

# === 综合测试网页 ===
@app.route("/debug")
def debug():
    return render_template("debug.html")

@app.route("/video_feed/debug/debug")
def debug_video_feed():
    key = "debug-debug"
    if key not in camera_instances:
        camera_instances[key] = SignCamera(target_sequence=[])  # 空任务序列
    return Response(camera_instances[key].generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status_feed/debug/debug")
def debug_status_feed():
    key = "debug-debug"
    if key in camera_instances:
        status = camera_instances[key].get_status()
        # 🟦 获取左右手方向（你应在 motiondetection.py 中加入 get_last_motion("left") 方法）
        status["motion"] = {
            "left": camera_instances[key].motion.get_last_motion("left"),
            "right": camera_instances[key].motion.get_last_motion("right")
        }
        return jsonify(status)
    return jsonify({"current": {}, "detected": [], "completed": False, "motion": {"left": "-", "right": "-"}})


# === 启动服务器 ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
