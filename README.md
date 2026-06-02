# CareBot AI — Fall Detection System

<img width="1280" height="1280" alt="image" src="https://github.com/user-attachments/assets/8a9ced80-248b-4c44-ab6a-429e49543b67" />
<img width="941" height="748" alt="image" src="https://github.com/user-attachments/assets/d691b49d-ce39-4d97-a28a-bb6f49c5d511" />
<img width="925" height="1599" alt="image" src="https://github.com/user-attachments/assets/7b8f496c-c69c-4606-b23e-733591615043" />

**Intelligent Real-Time Fall Detection System** using Computer Vision.

Combines **MediaPipe** and **YOLOv8-Pose** for high accuracy and reliability.

![CareBot AI](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

---

## ✨ Features

- **Dual Model Mode**: Runs MediaPipe + YOLOv8 Pose simultaneously
- **Advanced Posture Verification**: Uses body angles, aspect ratio, and velocity
- **Real-time Dashboard**: Beautiful monitoring interface
- **Complete Event Logging**: Video clips, screenshots, CSV, and HTML reports
- **Robust Performance**: Works in low light, noise, and varying conditions
- **Web Interface**: Access via browser (mobile friendly)

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- Ultralytics (YOLOv8)
- FastAPI + Uvicorn

---

## 🚀 Quick Start

1. Dual Model Mode (Recommended)
python main.py

2. Run Web Server Mode
Bashpython server.py
Then open your browser at: http://localhost:8000

3. Analysis Tools
conditions_test.py — Test system under different conditions

param_comparison.py — Compare different parameter settings

evaluator.py — Dual model evaluation

## 📁 Project Structure
imageProcessing/
├── main.py                    # Main dual-mode application

├── server.py                  # Web server

├── modules/

│   ├── detection_logic.py

│   ├── verification_logic.py

│   ├── logger_utils.py

│   ├── dashboard.py

│   ├── alert_system.py

│   ├── image_processor.py

│   └── report_generator.py

├── output/                    # Generated reports, videos & screenshots

├── events_log.csv             # All events log

├── analytics.html             # Analytics dashboard

├── yolov8n-pose.pt            # YOLOv8 Pose model

└── pose_landmarker.task       # MediaPipe model

## 📊 Automatic Reports
The system automatically generates:

Professional HTML session reports
Pre & post fall video recordings
Multiple processed images (Original, Grayscale, Edge Detection, Enhanced)
Structured CSV logs for further analysis
