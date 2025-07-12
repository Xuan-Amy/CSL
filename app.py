from flask import Flask
from flask_sock import Sock
from routes.main import main_bp
from routes.admin import admin_bp
from routes.websocket import websocket_sock
from config import warmup_hands

app = Flask(__name__)
sock = Sock(app)

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(admin_bp)
websocket_sock(sock)

# 启动 warmup
warmup_hands()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
