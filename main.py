import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import os
from datetime import datetime
from collections import deque

from modules.detection_logic import PersonTracker
from modules.verification_logic import PostureVerifier
from modules.logger_utils import log_event, buffer_frame
from modules.dashboard import Dashboard
from modules.report_generator import generate_report
from modules.alert_system import play_fall_alert


def get_video_source():
    print("\n--- CareBot AI System Settings ---")
    print("1. Live Camera Stream")
    print("2. Analyze Recorded Video File")
    choice = input("Select input source (1 or 2): ").strip()
    
    if choice == '1':
        print("[Status] Live Camera Mode")
        return 0
    elif choice == '2':
        raw_path = input("Enter video file path: ").strip('"').strip("'").strip()
        video_path = os.path.normpath(raw_path)
        if os.path.exists(video_path):
            print(f"[Status] Video loaded: {video_path}")
            return video_path
        else:
            print(f"[Error] File not found!")
            return None


def start_engine():
    print("\n=== CareBot AI - Dual Model Mode (MediaPipe + YOLO) ===\n")

    source = get_video_source()
    if source is None:
        return

    view = cv2.VideoCapture(source)
    if not view.isOpened():
        print("[Error] Cannot open video source.")
        return

    # === Create Trackers with clear separation ===
    tracker_mp = PersonTracker(max_persons=3)
    tracker_mp.USE_YOLO = False   # Force MediaPipe

    tracker_yolo = PersonTracker(max_persons=3)
    tracker_yolo.USE_YOLO = True  # Force YOLO

    print(f"[Info] MediaPipe Tracker initialized")
    print(f"[Info] YOLO Tracker initialized (USE_YOLO = {tracker_yolo.USE_YOLO})")

    verifiers_mp = {}
    verifiers_yolo = {}
    dashboard = Dashboard(width=460, height=720)

    # Video Export
    os.makedirs('output', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join('output', f'annotated_dual_{ts}.mp4')
    out_w = int(view.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(view.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = view.get(cv2.CAP_PROP_FPS) or 20
    video_writer = cv2.VideoWriter(export_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_w, out_h))

    session_start = datetime.now()
    frame_count = 0
    fall_count_mp = 0
    fall_count_yolo = 0
    fall_timeline = []
    fps_history = deque(maxlen=60)
    last_time = datetime.now()
    PROCESS_EVERY_N = 2

    last_fall_mp = {}
    last_fall_yolo = {}

    print("[System] Dual Mode Started. Press 'q' to stop.\n")

    while True:
        success, img = view.read()
        if not success:
            break

        frame_count += 1
        buffer_frame(img)
        should_process = (frame_count % PROCESS_EVERY_N == 0)

        now = datetime.now()
        delta = (now - last_time).total_seconds()
        last_time = now
        if delta > 0:
            fps_history.append(1.0 / delta)

        persons_mp = persons_yolo = []
        any_fall_mp = any_fall_yolo = False
        max_conf_mp = max_conf_yolo = 0.0

        if should_process:
            # MediaPipe
            persons_mp = tracker_mp.get_persons(img)
            for p in persons_mp:
                pid = p['id']
                if pid not in verifiers_mp:
                    verifiers_mp[pid] = PostureVerifier(confirmation_frames=3)
                is_fall, conf = verifiers_mp[pid].evaluate_posture(p['box'], p['landmarks'])
                p['is_fall'] = is_fall
                p['confidence'] = conf
                if is_fall and not last_fall_mp.get(pid, False):
                    fall_count_mp += 1
                    play_fall_alert()
                last_fall_mp[pid] = is_fall
                if is_fall:
                    any_fall_mp = True
                    max_conf_mp = max(max_conf_mp, conf)

            # YOLO
            persons_yolo = tracker_yolo.get_persons(img)
            for p in persons_yolo:
                pid = p['id']
                if pid not in verifiers_yolo:
                    verifiers_yolo[pid] = PostureVerifier(confirmation_frames=3)
                is_fall, conf = verifiers_yolo[pid].evaluate_posture(p['box'], p['landmarks'])
                p['is_fall'] = is_fall
                p['confidence'] = conf
                if is_fall and not last_fall_yolo.get(pid, False):
                    fall_count_yolo += 1
                    play_fall_alert()
                last_fall_yolo[pid] = is_fall
                if is_fall:
                    any_fall_yolo = True
                    max_conf_yolo = max(max_conf_yolo, conf)

        img = draw_dual_ui(img, persons_mp, persons_yolo, frame_count, fall_count_mp, fall_count_yolo)
        video_writer.write(img)

        fall_timeline.append(1 if (any_fall_mp or any_fall_yolo) else 0)

        dashboard.update(
            is_fall=any_fall_mp or any_fall_yolo,
            fall_count_mp=fall_count_mp,
            fall_count_yolo=fall_count_yolo,
            frame_count=frame_count,
            conf_mp=max_conf_mp,
            conf_yolo=max_conf_yolo
        )

        cv2.imshow('CareBot AI - Dual Mode', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_writer.release()
    view.release()
    dashboard.close()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

    print("\n" + "="*80)
    print("                    FINAL COMPARISON")
    print("="*80)
    print(f"MediaPipe Falls : {fall_count_mp}")
    print(f"YOLOv8 Falls    : {fall_count_yolo}")
    print(f"Total Frames    : {frame_count}")
    print(f"Avg FPS         : {avg_fps:.2f}")
    print("="*80)


def draw_dual_ui(img, persons_mp, persons_yolo, frame_count, fall_mp, fall_yolo):
    h, w = img.shape[:2]
    for p in persons_mp:
        x, y, bw, bh = p['box']
        color = (0, 255, 0) if p.get('is_fall', False) else (80, 200, 120)
        cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 3)
        cv2.putText(img, f"MP{p['id']+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for p in persons_yolo:
        x, y, bw, bh = p['box']
        color = (0, 0, 255) if p.get('is_fall', False) else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(img, f"YOLO{p['id']+1}", (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(img, f"Frame: {frame_count} | MP: {fall_mp} | YOLO: {fall_yolo}", 
                (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return img


if __name__ == "__main__":
    start_engine()