# Intelligent Football Performance Analysis Through Video Analytics

A state-of-the-art computer vision pipeline for automated player/ball tracking, team classification, perspective homography mapping (screen pixels to real-world 2D pitch coordinates), kinematic metrics calculation (speed, distance, heatmaps), and split-screen tactical visualization from uncalibrated single-camera video streams.

---
  
## Architecture Overview

```
Broadcast Video Feed -> Pre-processing -> YOLOv8 Detection -> ByteTrack Tracking
                                              ↓                        ↓
2D Pitch Template -> Perspective Homography Transform -> Kinematic Metrics Engine
                                              ↓                        ↓
                                  Dual-View Video Rendering & Analytics Export
```

---

## Features
- **Object Detection & Multi-Object Tracking**: Detects players, referees, and the ball using YOLOv8, and tracks entities reliably using ByteTrack.
- **Team Assignment**: Automatically clusters players into Team 1 vs Team 2 using HSV jersey color K-Means segmentation.
- **Dynamic Perspective Homography**: Converts foot position pixels `(x, y)` to standard FIFA 105m x 68m tactical pitch metric coordinates `(X_meters, Y_meters)`.
- **Kinematic Metrics**: Computes instantaneous speed (km/h), smoothed via moving window filter, and cumulative distance covered (meters).
- **Dual-View Rendering**: Generates split-screen videos featuring broadcast frame annotations alongside a top-down 2D tactical pitch view with live player markers and trailing movement paths.
- **Metrics Export**: Exports structured metrics as CSV for match reports.

---

## Project Structure

```
Project/
├── .venv/                         # Isolated Python Virtual Environment
├── configs/
│   └── pitch_config.json          # Standard 105x68m pitch geometry & homography keypoints
├── data/
│   ├── input/                     # Raw input video clips
│   ├── output/                    # Annotated videos & exported CSVs
│   └── pitch_templates/           # 2D pitch background graphics
├── models/                        # Pre-trained YOLO weights
├── src/
│   ├── perception/                # Detector, Tracker, Team Assigner
│   ├── geometry/                  # Homography Transformer & Pitch Template Generator
│   ├── analytics/                 # Kinematic Metrics & Trajectories
│   ├── visualization/             # Split-screen & Pitch Drawers
│   └── utils/                     # Video I/O helpers
├── main.py                        # Central CLI Pipeline Entry Point
├── test_pipeline.py               # End-to-end synthetic test harness
├── requirements.txt               # Dependencies
├── PROJECT_CONTEXT.md             # Project Context & Rules
├── DEVELOPMENT_LOG.md             # Log of features and decisions
├── CURRENT_STATE.md               # Live project state
├── NEXT_TASKS.md                  # Remaining roadmap tasks
└── README.md                      # Usage documentation
```

---

## Quick Start Guide

### 1. Environment Setup
Activate the virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Or install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Synthetic Test Harness
Verify end-to-end pipeline execution with auto-generated synthetic video:
```bash
python test_pipeline.py
```

### 3. Run Pipeline on Custom Match Video
```bash
python main.py --input data/input/your_match_clip.mp4 --output data/output/annotated_match.mp4 --csv data/output/match_metrics.csv
```
