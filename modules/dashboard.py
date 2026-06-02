import cv2
import numpy as np
from datetime import datetime
from collections import deque


class Dashboard:
    def __init__(self, width=460, height=720):
        self.width = width
        self.height = height
        self.window_name = 'CareBot AI - Dual Model Dashboard'

        self.BG = (30, 30, 30)
        self.CARD_BG = (45, 45, 45)
        self.GREEN = (80, 200, 120)
        self.RED = (60, 60, 220)
        self.YELLOW = (0, 200, 220)
        self.WHITE = (220, 220, 220)
        self.MUTED = (130, 130, 130)
        self.ACCENT = (200, 140, 80)
        self.MP_COLOR = (80, 220, 80)
        self.YOLO_COLOR = (60, 140, 255)

        self.fps_history = deque(maxlen=30)
        self.last_time = datetime.now()
        self.fall_timeline = deque(maxlen=200)
        self.session_start = datetime.now()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)

    def _draw_card(self, canvas, x, y, w, h, label, value, value_color=None):
        if value_color is None:
            value_color = self.WHITE
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.CARD_BG, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (70,70,70), 1)
        cv2.putText(canvas, label, (x+12, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)
        cv2.putText(canvas, str(value), (x+12, y+h-12), cv2.FONT_HERSHEY_DUPLEX, 0.75, value_color, 2)

    def update(self, is_fall, fall_count_mp, fall_count_yolo, frame_count, conf_mp=0.0, conf_yolo=0.0):
        now = datetime.now()
        delta = (now - self.last_time).total_seconds()
        self.last_time = now
        fps = 1.0 / delta if delta > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        self.fall_timeline.append(1 if is_fall else 0)

        canvas = np.full((self.height, self.width, 3), self.BG, dtype=np.uint8)

        cv2.putText(canvas, "CareBot AI - Dual Mode", (14, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, self.ACCENT, 2)
        cv2.putText(canvas, "MediaPipe vs YOLOv8", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)
        cv2.line(canvas, (14, 60), (self.width-14, 60), (70,70,70), 1)

        # Status
        status_text = "!! FALL DETECTED !!" if is_fall else "Normal"
        status_color = self.RED if is_fall else self.GREEN
        cv2.rectangle(canvas, (14, 68), (self.width-14, 118), self.CARD_BG, -1)
        cv2.rectangle(canvas, (14, 68), (self.width-14, 118), (60,60,220) if is_fall else (40,100,60), 2)
        cv2.putText(canvas, "Status", (26, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)
        cv2.putText(canvas, status_text, (26, 110), cv2.FONT_HERSHEY_DUPLEX, 0.72, status_color, 2)

        cw = (self.width - 36) // 2
        ch = 78

        self._draw_card(canvas, 14, 126, cw, ch, "MediaPipe Falls", fall_count_mp, self.MP_COLOR)
        self._draw_card(canvas, 22+cw, 126, cw, ch, "YOLOv8 Falls", fall_count_yolo, self.YOLO_COLOR)

        total = fall_count_mp + fall_count_yolo
        self._draw_card(canvas, 14, 126+ch+8, cw, ch, "Total Falls", total, self.RED if total > 0 else self.WHITE)
        self._draw_card(canvas, 22+cw, 126+ch+8, cw, ch, "FPS", f"{avg_fps:.1f}", self.YELLOW)

        elapsed = int((now - self.session_start).total_seconds())
        mins, secs = divmod(elapsed, 60)
        self._draw_card(canvas, 14, 126+(ch+8)*2, cw, ch, "Frames", frame_count, self.WHITE)
        self._draw_card(canvas, 22+cw, 126+(ch+8)*2, cw, ch, "Time", f"{mins:02d}:{secs:02d}", self.ACCENT)

        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyWindow(self.window_name)