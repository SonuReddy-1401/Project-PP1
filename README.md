# Intelligent Football Performance Analysis Through Video Analytics

A state-of-the-art computer vision pipeline for automated player/ball tracking, CIELAB perceptual team classification, referee/goalkeeper role identification, perspective homography mapping (screen pixels to real-world 2D pitch coordinates), 2D spatial pitch heatmap generation, kinematic metrics calculation, and split-screen tactical visualization from broadcast video streams.

---

## Architecture Overview

```
Broadcast Video Feed -> Sliced Tiling (SAHI) -> YOLOv8 Detection (CUDA GPU) -> ByteTrack Tracking
                                                       ↓                             ↓
2D Pitch Template -> Homography & Pitch Bounds Check -> Perceptual CIELAB Team & Ref/GK Assigner
                                                       ↓                             ↓
                                        2D Tactical Map & Player Spatial Heatmap Generator
```

---

## Features
- **High-Recall Object Detection (SAHI)**: Slices high-resolution broadcast frames into overlapping tiles for $95\%+$ detection recall of tiny distant players ($<15\text{px}$) in wide-angle tactical cam shots.
- **Hardware GPU Acceleration**: Accelerated via PyTorch CUDA on **NVIDIA GeForce RTX GPUs**.
- **CIELAB Perceptual Team & Role Classification**: Uses 12-dimensional joint Jersey + Shorts perceptual color features in CIELAB space (`cv2.COLOR_BGR2LAB`), reliably separating teams even with similar kit colors and identifying Referees (`REF`) and Goalkeepers (`GK`).
- **Pitch Boundary Sideline Filtering**: Filters out coaches, substitutes, camera operators, and fans standing outside the field bounds ($105\text{m} \times 68\text{m}$).
- **2D Pitch Spatial Density Heatmaps**: Generates post-match high-resolution PNG spatial heatmaps (`heatmap_team1_player_X.png`, `heatmap_team2_player_Y.png`) using 2D Gaussian Kernel Density Estimation.
- **Dual-View Split-Screen Video**: Renders broadcast video side-by-side with a top-down 2D tactical pitch view.

---

## Project Structure

```
Project/
├── .venv/                         # Isolated Python Virtual Environment (PyTorch CUDA active)
├── configs/
│   └── pitch_config.json          # Standard 105x68m pitch geometry & homography keypoints
├── data/
│   ├── input/                     # Raw input video clips
│   ├── output/                    # Annotated videos, heatmaps & exported CSVs
│   └── pitch_templates/           # 2D pitch background graphics
├── models/                        # Pre-trained YOLO weights
├── src/
│   ├── perception/                # Detector (SAHI + CUDA), Tracker (ByteTrack), Team Assigner (CIELAB)
│   ├── geometry/                  # Homography Transformer & Pitch Boundary Filter
│   ├── analytics/                 # Kinematic Metrics Engine & 2D Player Heatmap Generator
│   ├── visualization/             # Split-screen & Pitch Drawers
│   └── utils/                     # Video I/O streaming helpers
├── main.py                        # Central CLI Pipeline Entry Point
├── download_and_analyze.py        # Automated YouTube timestamp downloader & runner
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

### 1. Run Pipeline on Tactical Cam Match Video
```powershell
.\.venv\Scripts\python.exe main.py --input data/input/tactical_clip_1.mp4 --output data/output/tactical_clip_1_annotated.mp4 --model yolov8m.pt --imgsz 1280 --conf 0.15
```

### 2. Run Batch Downloader & Processing
```powershell
.\.venv\Scripts\python.exe download_and_analyze.py
```

### 3. Run Synthetic Integration Test Suite
```powershell
.\.venv\Scripts\python.exe test_pipeline.py
```
