import json, time
import numpy as np
import cv2
from config import QUESTIONS, camera_instances
from camera_ws import SignRecognizerWS


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
            ws.send(json.dumps({"error": f"识别错误：{str(e)}"}))
            break

def websocket_sock(sock):
    @sock.route("/ws/recognize")
    def recognize_general(ws):
        recognizer = SignRecognizerWS([])
        handle_ws(ws, recognizer)

    @sock.route("/ws/recognize/<level>/<qid>")
    def recognize_task(ws, level, qid):
        question = QUESTIONS.get(level, {}).get(qid)
        if not question:
            ws.send(json.dumps({"error": "题目不存在"}))
            return
        recognizer = SignRecognizerWS(question.get("target_sequence", []))
        camera_instances[f"{level}-{qid}"] = recognizer
        handle_ws(ws, recognizer)

    @sock.route("/ws/recognize/debug/debug")
    def recognize_debug(ws):
        recognizer = SignRecognizerWS([])
        camera_instances["debug-debug"] = recognizer
        handle_ws(ws, recognizer)
