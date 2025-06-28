import numpy as np

class MotionDetector:
    def __init__(self, max_len=10, threshold=0.05, size_threshold=0.02):
        self.max_len = max_len
        self.threshold = threshold
        self.size_threshold = size_threshold

        self.left_x_queue = []
        self.left_y_queue = []
        self.left_size_queue = []

        self.right_x_queue = []
        self.right_y_queue = []
        self.right_size_queue = []

    def update(self, left_hand=None, right_hand=None):
        """
        更新左右手的关键点数据。

        参数:
        - left_hand: Tuple(wrist, index_tip, thumb_tip) or None
        - right_hand: Tuple(wrist, index_tip, thumb_tip) or None
        """
        if left_hand:
            wrist, index_tip, thumb_tip = left_hand
            self.left_x_queue.append(wrist.x)
            self.left_y_queue.append(wrist.y)
            dist1 = np.sqrt((wrist.x - index_tip.x)**2 + (wrist.y - index_tip.y)**2)
            dist2 = np.sqrt((wrist.x - thumb_tip.x)**2 + (wrist.y - thumb_tip.y)**2)
            hand_size = (dist1 + dist2) / 2
            self.left_size_queue.append(hand_size)

            if len(self.left_x_queue) > self.max_len:
                self.left_x_queue.pop(0)
                self.left_y_queue.pop(0)
                self.left_size_queue.pop(0)

        if right_hand:
            wrist, index_tip, thumb_tip = right_hand
            self.right_x_queue.append(wrist.x)
            self.right_y_queue.append(wrist.y)
            dist1 = np.sqrt((wrist.x - index_tip.x)**2 + (wrist.y - index_tip.y)**2)
            dist2 = np.sqrt((wrist.x - thumb_tip.x)**2 + (wrist.y - thumb_tip.y)**2)
            hand_size = (dist1 + dist2) / 2
            self.right_size_queue.append(hand_size)

            if len(self.right_x_queue) > self.max_len:
                self.right_x_queue.pop(0)
                self.right_y_queue.pop(0)
                self.right_size_queue.pop(0)

    def has_moved(self, hand: str, direction: str) -> bool:
        """
        检查某只手是否朝指定方向移动。
        
        参数:
        - hand: 'left' 或 'right'
        - direction: 'right', 'left', 'up', 'down', 'closer', 'farther'
        """
        if hand == 'left':
            x_queue = self.left_x_queue
            y_queue = self.left_y_queue
            size_queue = self.left_size_queue
        elif hand == 'right':
            x_queue = self.right_x_queue
            y_queue = self.right_y_queue
            size_queue = self.right_size_queue
        else:
            raise ValueError("hand 必须是 'left' 或 'right'")

        if len(x_queue) < self.max_len:
            return False

        half = self.max_len // 2
        x_before = np.mean(x_queue[:half])
        x_after = np.mean(x_queue[half:])
        y_before = np.mean(y_queue[:half])
        y_after = np.mean(y_queue[half:])
        size_before = np.mean(size_queue[:half])
        size_after = np.mean(size_queue[half:])

        if direction == "right":
            return x_after - x_before > self.threshold
        elif direction == "left":
            return x_before - x_after > self.threshold
        elif direction == "down":
            return y_after - y_before > self.threshold
        elif direction == "up":
            return y_before - y_after > self.threshold
        elif direction == "closer":
            return size_after - size_before > self.size_threshold
        elif direction == "farther":
            return size_before - size_after > self.size_threshold
        else:
            raise ValueError("方向必须是 'right', 'left', 'up', 'down', 'closer', 'farther'")

    def clear(self):
        self.left_x_queue.clear()
        self.left_y_queue.clear()
        self.left_size_queue.clear()
        self.right_x_queue.clear()
        self.right_y_queue.clear()
        self.right_size_queue.clear()

    def get_last_motion(self, hand: str) -> str:
        if hand == "left":
            size_queue = self.left_size_queue
        elif hand == "right":
            size_queue = self.right_size_queue
        else:
            return "-"

        if len(size_queue) < self.max_len:
            return "-"

        half = self.max_len // 2
        size_before = np.mean(size_queue[:half])
        size_after = np.mean(size_queue[half:])

        # ✅ 优先判断尺寸变化，判断靠近或远离
        size_diff = size_after - size_before
        if size_diff > self.size_threshold * 2:
            return "closer"
        elif size_diff < -self.size_threshold * 2:
            return "farther"

        # 否则继续判断方向移动
        directions = ["right", "left", "up", "down"]
        for dir in directions:
            if self.has_moved(hand, dir):
                return dir

        return "-"


