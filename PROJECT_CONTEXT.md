# Project Context: Intelligent Football Performance Analysis Through Video Analytics

## Overview
This project is an advanced computer-vision sports analytics platform designed to convert single-camera, uncalibrated broadcast or local football match footage into high-fidelity spatiotemporal metrics (X, Y real-world tactical coordinates, speed, distance covered, acceleration, heatmaps, and event detection).

## Active Project Root
- **Path**: `C:\CLG\PP1\Project - PP1`

## Core Technical Challenges Solved
1. **Dynamic Camera Homography**: Converting screen pixel coordinates \((x, y)\) from a panning/zooming broadcast camera into standard 2D pitch meter coordinates \((X, Y)\).
2. **Severe Occlusion & Identity Persistence**: Tracking players continuously through crowding and overlapping using YOLO detection and ByteTrack tracking.
3. **Team Classification**: Jersey color clustering using HSV color space and K-Means segmentation.
4. **Kinematic Metrics**: Velocity smoothing via moving window filtering to eliminate pixel projection jitter.

## System Architecture

```
Broadcast Video Feed -> Pre-processing -> YOLOv8 Detection -> ByteTrack Tracking
                                              ↓                        ↓
2D Pitch Template -> Perspective Homography Transform -> Kinematic Metrics Engine
                                              ↓                        ↓
                                  Dual-View Video Rendering & Analytics Export
```

## Modular Directory Structure
```
C:\CLG\PP1\Project - PP1/
├── .venv/                         # Isolated Python Virtual Environment
├── configs/
│   └── pitch_config.json          # Standard 105x68m pitch geometry & homography keypoints
├── data/
│   ├── input/                     # Raw input video clips
│   ├── output/                    # Annotated videos & exported CSVs
│   └── pitch_templates/           # 2D pitch background graphics
├── models/                        # Pre-trained YOLO weights
├── src/
│   ├── perception/                # Detector (YOLOv8), Tracker (ByteTrack), Team Assigner (K-Means)
│   ├── geometry/                  # Homography Transformer & 2D Tactical Pitch Template Generator
│   ├── analytics/                 # Kinematic Metrics & Trajectory smoothing
│   ├── visualization/             # Dual-view split-screen & Pitch Drawers
│   └── utils/                     # Video I/O helpers
├── main.py                        # Central CLI Pipeline Entry Point
├── test_pipeline.py               # End-to-end synthetic test harness
├── requirements.txt               # Pinned Dependencies
├── PROJECT_CONTEXT.md             # Project Context & Architectural Guidelines
├── DEVELOPMENT_LOG.md             # Log of features and migration decisions
├── CURRENT_STATE.md               # Live project state
├── NEXT_TASKS.md                  # Roadmap & Remaining tasks
└── README.md                      # Complete usage & setup instructions
```

## Technology Stack
- **Python**: 3.13.12
- **Object Detection**: `ultralytics` (YOLOv8)
- **Object Tracking**: `supervision` / `ByteTrack`
- **Computer Vision & Geometry**: `opencv-python`
- **Analytics & Math**: `numpy`, `pandas`, `scipy`, `scikit-learn`
- **Visuals & Plotting**: `matplotlib`, `plotly`
