import cv2
import numpy as np
from collections import deque
import os
import urllib.request

USE_YOLO = True
YOLO_MODEL = "yolov8n-pose.pt"


class PersonTracker:
    def __init__(self, max_persons=3, smooth_window=25):
        self.max_persons = max_persons
        self.smooth_window = smooth_window
        self.frame_index = 0
        self.box_histories = {}
        self.last_boxes = {}
        self.next_id = 0

        if USE_YOLO:
            self._init_yolo()
        else:
            self._init_mediapipe()

    def _init_mediapipe(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        MODEL_PATH = "pose_landmarker.task"
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

        if not os.path.exists(MODEL_PATH):
            print("[Setup] Downloading MediaPipe model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[Setup] Model downloaded!")

        os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.max_persons,
            min_pose_detection_confidence=0.5,
        )
        self.pose_analyzer = vision.PoseLandmarker.create_from_options(options)
        self.is_yolo = False
        print("[Detection] MediaPipe Pose (CPU Mode)")

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
            print(f"[YOLO] Loading {YOLO_MODEL} ...")
            self.model = YOLO(YOLO_MODEL)
            self.is_yolo = True
            print("[Detection] YOLOv8 Pose Loaded")
        except Exception as e:
            print(f"[ERROR] YOLO failed: {e}")
            self.is_yolo = False
            self._init_mediapipe()

    def _iou(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def _match_ids(self, raw_boxes):
        matched = {}
        used_raw = set()
        for pid, last_box in list(self.last_boxes.items()):
            best_iou, best_i = 0, -1
            for i, rb in enumerate(raw_boxes):
                if i in used_raw:
                    continue
                iou = self._iou(last_box, rb)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou > 0.25 and best_i >= 0:
                matched[pid] = raw_boxes[best_i]
                used_raw.add(best_i)

        for i, rb in enumerate(raw_boxes):
            if i not in used_raw:
                matched[self.next_id] = rb
                self.next_id += 1

        self.last_boxes = matched
        return matched

    def _smooth_box(self, pid, box):
        if pid not in self.box_histories:
            self.box_histories[pid] = deque(maxlen=self.smooth_window)

        hist = self.box_histories[pid]
        hist.append(box)

        if len(hist) < 5:
            return [int(b) for b in box]

        smoothed = []
        alpha = 0.68   # Higher = more stability

        for i in range(4):  # x, y, w, h
            values = [b[i] for b in hist]
            
            # Exponential Smoothing
            smoothed_val = values[-1]
            for v in reversed(values[:-1]):
                smoothed_val = alpha * smoothed_val + (1 - alpha) * v

            # Extra stabilization for width & height
            if i >= 2:
                smoothed_val = round(smoothed_val / 10) * 10

            smoothed.append(int(smoothed_val))

        return smoothed

    def get_persons(self, frame):
        h, w = frame.shape[:2]

        if getattr(self, 'is_yolo', False):
            return self._get_persons_yolo(frame, w, h)
        else:
            return self._get_persons_mediapipe(frame, w, h)

    def _get_persons_mediapipe(self, frame, w, h):
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self.frame_index * 33
        self.frame_index += 1

        result = self.pose_analyzer.detect_for_video(mp_img, ts)

        if not result.pose_landmarks:
            self.last_boxes = {}
            return []

        raw_boxes = []
        raw_lms = []
        for lms in result.pose_landmarks:
            box = self._landmarks_to_box(lms, w, h)
            if box:
                raw_boxes.append(box)
                raw_lms.append(lms)

        id_to_box = self._match_ids(raw_boxes)
        return self._build_persons(id_to_box, raw_boxes, raw_lms)

    def _landmarks_to_box(self, pts, w, h):
        vx = [p.x for p in pts if p.visibility > 0.5]
        vy = [p.y for p in pts if p.visibility > 0.5]
        if len(vx) < 5:
            return None
        return [
            max(0, int(min(vx) * w) - 20),
            max(0, int(min(vy) * h) - 20),
            min(w, int((max(vx) - min(vx)) * w) + 40),
            min(h, int((max(vy) - min(vy)) * h) + 40)
        ]

    def _get_persons_yolo(self, frame, w, h):
        results = self.model(frame, conf=0.3, verbose=False)
        raw_boxes = []
        raw_kpts = []

        for r in results:
            if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xywh.cpu().numpy()
            kpts = r.keypoints.xy.cpu().numpy()
            for box, kpt in zip(boxes, kpts):
                cx, cy, bw, bh = box
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                raw_boxes.append([max(0, x1), max(0, y1), int(bw), int(bh)])
                raw_kpts.append(kpt)

        if not raw_boxes:
            self.last_boxes = {}
            return []

        id_to_box = self._match_ids(raw_boxes)
        return self._build_persons_yolo(id_to_box, raw_boxes, raw_kpts)

    def _build_persons(self, id_to_box, raw_boxes, raw_lms):
        persons = []
        colors = [(80, 200, 120), (60, 140, 220), (60, 60, 220), (0, 200, 220)]
        for pid, box in id_to_box.items():
            best_lms = None
            best_iou = 0
            for lms, rb in zip(raw_lms, raw_boxes):
                iou = self._iou(box, rb)
                if iou > best_iou:
                    best_iou = iou
                    best_lms = lms
            if best_lms:
                smooth_box = self._smooth_box(pid, box)
                persons.append({
                    'id': pid,
                    'box': smooth_box,
                    'landmarks': best_lms,
                    'color': colors[pid % 4]
                })
        return persons

    def _build_persons_yolo(self, id_to_box, raw_boxes, raw_kpts):
        persons = []
        colors = [(80, 200, 120), (60, 140, 220), (60, 60, 220), (0, 200, 220)]
        for pid, box in id_to_box.items():
            best_kpt = None
            best_iou = 0
            for kpt, rb in zip(raw_kpts, raw_boxes):
                iou = self._iou(box, rb)
                if iou > best_iou:
                    best_iou = iou
                    best_kpt = kpt
            if best_kpt is not None:
                smooth_box = self._smooth_box(pid, box)
                persons.append({
                    'id': pid,
                    'box': smooth_box,
                    'landmarks': best_kpt,
                    'color': colors[pid % 4]
                })
        return persons