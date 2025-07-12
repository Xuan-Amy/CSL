from flask import Blueprint, render_template, request, redirect
from config import QUESTIONS, QUESTIONS_FILE, USER_FILE
from utils.io_utils import load_json, save_json
import os

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/admin")
def admin():
    return render_template("admin.html", questions=QUESTIONS, users=load_json(USER_FILE))

@admin_bp.route("/admin/add_user", methods=["POST"])
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

@admin_bp.route("/admin/delete_user/<username>")
def delete_user(username):
    users = load_json(USER_FILE)
    users.pop(username, None)
    save_json(USER_FILE, users)
    return redirect("/admin")

@admin_bp.route("/admin/questions")
def admin_questions():
    return render_template("admin_questions.html", questions=QUESTIONS)

@admin_bp.route("/admin/add_question", methods=["POST"])
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

@admin_bp.route("/admin/delete_question/<level>/<qid>")
def delete_question(level, qid):
    if level in QUESTIONS and qid in QUESTIONS[level]:
        del QUESTIONS[level][qid]
        if not QUESTIONS[level]:
            del QUESTIONS[level]
        save_json(QUESTIONS_FILE, QUESTIONS)
    return redirect("/admin/questions")

@admin_bp.route("/admin/edit_question/<level>/<qid>", methods=["GET"])
def edit_question_get(level, qid):
    q = QUESTIONS.get(level, {}).get(qid)
    if not q:
        return "题目不存在", 404
    words = ",".join([i["word"] for i in q["target_sequence"]])
    motions = ",".join([i["motion"] or "" for i in q["target_sequence"]])
    return render_template("edit_question.html", level=level, qid=qid, question=q, words=words, motions=motions)

@admin_bp.route("/admin/edit_question/<level>/<qid>", methods=["POST"])
def edit_question_post(level, qid):
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
