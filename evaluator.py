import cv2
import os
import json
from datetime import datetime
from modules.detection_logic import PersonTracker
from modules.verification_logic import PostureVerifier


def run_evaluation(video_path: str, output_dir: str = 'output'):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Evaluator] Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    print("\n" + "="*70)
    print("     CareBot AI — Dual Model Evaluator (MediaPipe vs YOLO)")
    print("="*70)
    print(" SPACE → Mark current frame as FALL (Ground Truth)")
    print("   Q   → Finish and compute comparison")
    print("   P   → Pause / Resume")
    print("="*70 + "\n")

    # ====================== MODELS SETUP ======================
    models = {}
    for name in ["MediaPipe", "YOLO"]:
        det = PersonTracker(max_persons=3)
        det.USE_YOLO = (name == "YOLO")
        models[name] = {
            "tracker": det,
            "verifiers": {},
            "pred_labels": {},
            "prev_fall_states": {}
        }
        print(f"✅ {name} loaded")

    gt_labels = {}        # Ground Truth
    frame_count = 0
    paused = False

    cv2.namedWindow('CareBot Dual Evaluator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CareBot Dual Evaluator', 1000, 600)

    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1

            display = frame.copy()

            # Run both models
            for model_name, m in models.items():
                any_fall = False
                try:
                    if frame_count % 2 == 0:
                        persons = m["tracker"].get_persons(frame)
                        for p in persons:
                            pid = p['id']
                            if pid not in m["verifiers"]:
                                m["verifiers"][pid] = PostureVerifier(confirmation_frames=5)
                            
                            is_fall, _ = m["verifiers"][pid].evaluate_posture(p['box'], p['landmarks'])
                            if is_fall:
                                any_fall = True
                                m["pred_labels"][frame_count] = 1

                        m["prev_fall_states"] = {p['id']: any_fall for p in persons}
                except:
                    pass

                # Draw on screen
                color = (0, 0, 255) if any_fall else (0, 255, 0)
                cv2.putText(display, f"{model_name}: {'FALL' if any_fall else 'Normal'}", 
                          (20, 60 + 40 * list(models.keys()).index(model_name)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Ground Truth indicator
            if frame_count in gt_labels:
                cv2.putText(display, "GT: FALL ✓", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Progress
            progress = frame_count / total_frames if total_frames > 0 else 0
            cv2.rectangle(display, (0, total_frames-10 if total_frames > 0 else 470), 
                         (int(1000*progress), 480), (0,165,255), -1)

            cv2.putText(display, f"Frame: {frame_count}/{total_frames} | GT Falls: {len(gt_labels)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            cv2.imshow('CareBot Dual Evaluator', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            gt_labels[frame_count] = 1
            print(f"[GT] Frame #{frame_count} marked as FALL")

    cap.release()
    cv2.destroyAllWindows()

    # ====================== COMPUTE METRICS ======================
    all_frames = set(range(1, frame_count + 1))
    results = {}

    print("\nCalculating metrics...")

    for model_name, m in models.items():
        pred = set(m["pred_labels"].keys())
        gt = set(gt_labels.keys())

        TP = len(pred & gt)
        FP = len(pred - gt)
        FN = len(gt - pred)
        TN = len(all_frames - pred - gt)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = (TP + TN) / frame_count if frame_count > 0 else 0

        results[model_name] = {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "gt_falls": len(gt),
            "pred_falls": len(pred)
        }

    # ====================== SAVE REPORT ======================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"eval_dual_{ts}.json")
    txt_path  = os.path.join(output_dir, f"eval_dual_{ts}.txt")

    with open(json_path, 'w') as f:
        json.dump({"video": os.path.basename(video_path), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "total_frames": frame_count, "results": results}, f, indent=2)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("CareBot AI - Dual Model Evaluation Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Total Frames: {frame_count}\n")
        f.write(f"Ground Truth Falls: {len(gt_labels)}\n\n")
        
        for model, r in results.items():
            f.write(f"\n{model.upper()}:\n")
            f.write(f"  Precision : {r['precision']*100:.2f}%\n")
            f.write(f"  Recall    : {r['recall']*100:.2f}%\n")
            f.write(f"  F1 Score  : {r['f1_score']*100:.2f}%\n")
            f.write(f"  Accuracy  : {r['accuracy']*100:.2f}%\n")
            f.write(f"  TP/FP/FN/TN: {r['TP']}/{r['FP']}/{r['FN']}/{r['TN']}\n")

    # Print Summary
    print("\n" + "="*70)
    print("              FINAL COMPARISON RESULTS")
    print("="*70)
    for model, r in results.items():
        print(f"\n🔹 {model.upper()}:")
        print(f"   F1 Score  : {r['f1_score']*100:.2f}%")
        print(f"   Precision : {r['precision']*100:.2f}%")
        print(f"   Recall    : {r['recall']*100:.2f}%")
        print(f"   Falls Detected: {r['pred_falls']}")

    print(f"\n📊 Reports saved:")
    print(f"   → {txt_path}")
    print(f"   → {json_path}")
    print("="*70)

    return results


if __name__ == "__main__":
    path = input("Enter video path: ").strip('"').strip("'").strip()
    if os.path.exists(path):
        run_evaluation(path)
    else:
        print("Video not found!")