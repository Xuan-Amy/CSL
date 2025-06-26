import numpy as np

class MotionDetector:
    def __init__(self, max_len=10, threshold=0.05):
        self.max_len = max_len
        self.threshold = threshold
        self.x_queue = []
        self.y_queue = []

    def update(self, wrist_landmark):
        self.x_queue.append(wrist_landmark.x)
        self.y_queue.append(wrist_landmark.y)
        if len(self.x_queue) > self.max_len:
            self.x_queue.pop(0)
            self.y_queue.pop(0)

    def has_moved(self, direction: str) -> bool:
        if len(self.x_queue) < self.max_len:
            return False
        half = self.max_len // 2
        x_before = np.mean(self.x_queue[:half])
        x_after = np.mean(self.x_queue[half:])
        y_before = np.mean(self.y_queue[:half])
        y_after = np.mean(self.y_queue[half:])
        if direction == "right":
            return x_after - x_before > self.threshold
        elif direction == "left":
            return x_before - x_after > self.threshold
        elif direction == "down":
            return y_after - y_before > self.threshold
        elif direction == "up":
            return y_before - y_after > self.threshold
        else:
            raise ValueError("方向必须是 'right', 'left', 'up', 'down'")

    def clear(self):
        self.x_queue.clear()
        self.y_queue.clear()
