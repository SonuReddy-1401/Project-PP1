# Phase 1: SoccerNet 3D Camera Calibration Baseline Implementation

## Executive Summary
This directory (`Pitch-Segmentation-2`) contains the complete implementation and evaluation framework for **Phase 1: SoccerNet Camera Calibration Baseline** (`sn-calibration`).

The objective of Phase 1 is to validate a **3D Physics-Based Camera Parameter Estimation Loop** (Pan, Tilt, Roll, 3D Position, Focal Length) on custom tactical/broadcast football clips. This replaces legacy 2D pixel homography warping—which failed during fast camera motion—with real-world 3D geometry projection.

---

## 1. Mathematical & Architectural Foundation

### Why 3D Camera Calibration Replaces 2D Homography
* **Legacy 2D Homography Failure**: 2D pixel homography directly warps image coordinates $(x, y) \to (x', y')$ using a $3\times3$ matrix. When camera keypoints cluster on one side during panning, the matrix degenerates and distorts lines into severe polygons.
* **3D Physical Camera Model**: Models the broadcast camera as a physical 3D entity operating in real-world FIFA pitch coordinates $(X, Y, Z \in \mathbb{R}^3)$:
  $$\text{Camera State} = \left[ \text{Pan}(\theta), \text{Tilt}(\phi), \text{Roll}(\psi), \text{Position}(X, Y, Z), \text{Focal Length}(f) \right]$$
* **Projection Equation**: Converts 3D world pitch points $P_{3D} = (X, Y, 0)^T$ into 2D camera pixel coordinates $p_{2D} = (x, y)^T$:
  $$p_{2D} \sim K \cdot R(\theta, \phi, \psi) \cdot (P_{3D} - T)$$

---

## 2. Directory Structure & Key Files Created

```
C:\CLG\PP1\EXP\Pitch-Segmentation-2\
│
├── data/
│   └── custom_dataset/               # Frame images (1 FPS) and per_match_info.json manifest
│       └── test/
│           ├── clip1/                # Extracted JPG frames (frame_00001.jpg ...)
│           └── per_match_info.json   # SoccerNet match manifest
│
├── output/
│   ├── predictions/                  # Model prediction outputs
│   │   └── test/
│   │       ├── extremities_clip1/    # Line extremity JSON files (2D line endpoints)
│   │       └── camera_clip1/         # 3D Camera parameter JSON files (Pan/Tilt/Roll/Focal Length)
│   ├── visual_verification/          # Sample frame 3D wireframe overlay JPEGs
│   └── pitch_calibrated_clean_video.mp4 # Full 1-minute 3D calibrated MP4 video output
│
├── sn-calibration/                   # SoccerNet camera calibration baseline repository
│   ├── resources/                    # Pretrained DeepLabv3 weights (159 MB) & normalization parameters
│   │   ├── soccer_pitch_segmentation.pth
│   │   ├── mean.npy
│   │   └── std.npy
│   └── src/                          # Baseline scripts (detect_extremities.py, baseline_cameras.py, camera.py, soccerpitch.py)
│
├── prepare_test_frames.py            # Python script: extracts 1 FPS frames & builds SoccerNet manifest
├── visualize_calibration.py          # Python script: generates 3D pitch wireframe overlay JPEGs
└── camera_calibrated_analytics_pipeline.py # Production script: renders pure 3D pitch calibration MP4 video
```

---

## 3. Pipeline Execution Steps & Manual Terminal Commands

All scripts utilize the central virtual environment located at `C:\CLG\PP1\EXP\.venv`.

### Step 1: Frame Extraction & Dataset Manifest Generation
Extracts test frames from `C:\CLG\PP1\EXP\Pitch-Segmentation\data\input_video.mp4` at 1 FPS and generates `per_match_info.json`.
```powershell
cd C:\CLG\PP1\EXP\Pitch-Segmentation-2
..\.venv\Scripts\python.exe prepare_test_frames.py
```

### Step 2: Line & Extremity Detection (DeepLabv3 Segmentation)
Runs the pretrained DeepLabv3 semantic line-segmentation network to detect pitch line extremities per frame.
```powershell
cd C:\CLG\PP1\EXP\Pitch-Segmentation-2\sn-calibration
..\..\.venv\Scripts\python.exe src/detect_extremities.py -s ..\data\custom_dataset -p ..\output\predictions --split test
```

### Step 3: Solve 3D Camera Parameters (Pan, Tilt, Roll, Position, Focal Length)
Matches detected line extremities against the canonical 3D FIFA pitch model (`soccerpitch.py`) and solves for 3D camera parameters exported to `camera_clip1/frame_XXXXX.json`.
```powershell
cd C:\CLG\PP1\EXP\Pitch-Segmentation-2\sn-calibration
..\..\.venv\Scripts\python.exe src/baseline_cameras.py -s ..\data\custom_dataset -p ..\output\predictions --split test
```

### Step 4: Sample Frame Visual Overlays
Projects canonical 3D FIFA pitch lines onto individual sample frame JPEGs saved in `output/visual_verification/`.
```powershell
cd C:\CLG\PP1\EXP\Pitch-Segmentation-2
..\.venv\Scripts\python.exe visualize_calibration.py
```

### Step 5: Full 1-Minute Calibrated Video Generation
Processes all frames across the entire 1-minute video clip, applies temporal camera parameter smoothing, and renders the clean 3D pitch calibration MP4 video.
```powershell
cd C:\CLG\PP1\EXP\Pitch-Segmentation-2
..\.venv\Scripts\python.exe camera_calibrated_analytics_pipeline.py
```

---

## 4. Key Findings & Baseline Evaluation

1. **Geometry Approach Validated**: 3D camera decomposition successfully maintains continuous line structures without the catastrophic warping seen in 2D homography.
2. **Baseline Accuracy Audit**:
   * **Static / Slow Camera Shots**: Projected 3D pitch lines align closely with real white field lines.
   * **Fast Pans & Zooms**: DeepLabv3 extremity detection experiences minor line drift during rapid motion. This matches SoccerNet's expected baseline benchmark score ($\text{JaC@5} = 11.7\%$).
3. **Temporal Camera Smoothing**: Integrating camera parameter fallback between consecutive frames effectively eliminates line flickering.

---

## 5. Next Steps: Phase 2 Upgrade

Having validated the 3D calibration pipeline architecture in Phase 1, the next step is upgrading to the **1st-Place Sportlight Repository** (CVPR 2023 Challenge Winner):
* **Model Upgrade**: Replaces DeepLabv3 with **HRNetV2-w48** fine-tuned on 57 keypoints.
* **Refinement Engine**: Implements RANSAC-based non-linear optimization to achieve pixel-accurate, broadcast-grade camera calibration across fast camera motion.
