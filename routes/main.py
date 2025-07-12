from flask import Blueprint, render_template
from config import QUESTIONS

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS)

@main_bp.route("/task/<level>/<q>")
def task(level, q):
    question = QUESTIONS.get(level, {}).get(q)
    if not question:
        return f"题目 {level}-{q} 不存在", 404
    return render_template("task.html", level=level, q=q, prompt=question["prompt"], questions=QUESTIONS)

@main_bp.route("/debug_ws")
def debug_ws():
    return render_template("debug_ws.html")
