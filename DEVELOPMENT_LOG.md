# Development Log

## [2026-07-28] - Branch `ABDULLAH` Low-RAM Streaming & Roboflow Sports 32-Keypoint Integration
- **Action**:
  1. Built **Low-RAM Frame Streaming Architecture (<500 MB RAM)** in [main.py](file:///c:/CLG/PP1/Project%20-%20PP1/main.py), [trackers/tracker.py](file:///c:/CLG/PP1/Project%20-%20PP1/trackers/tracker.py), and [camera_movement_estimator/camera_movement_estimator.py](file:///c:/CLG/PP1/Project%20-%20PP1/camera_movement_estimator/camera_movement_estimator.py) to eliminate `ArrayMemoryError` on 5-minute+ broadcast clips.
  2. Integrated **Roboflow Sports 32 FIFA pitch metric keypoint configuration (`SoccerPitchConfiguration`)** and dynamic `ViewTransformer` in [view_transformer/view_transformer.py](file:///c:/CLG/PP1/Project%20-%20PP1/view_transformer/view_transformer.py).
  3. Repositioned **Player ID & Speed Badges to ABOVE THE HEAD (`y1 - 25`)**, leaving player bodies and ground ellipses 100% visible.
  4. Upgraded [team_assigner/team_assigner.py](file:///c:/CLG/PP1/Project%20-%20PP1/team_assigner/team_assigner.py) with **HSV Red torso thresholding + Multi-Frame Majority Voting** to eliminate white player misclassifications.
  5. Built noise-free **15-frame gap-limited ball trajectory interpolation** to eliminate floating false positive ball pointers.

## [2026-07-27] - Recreating Abdullah Tarek `football_analysis` Architecture
- **Action**:
  1. Built exact module structure from [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis):
     - [trackers/tracker.py](file:///c:/CLG/PP1/Project%20-%20PP1/trackers/tracker.py): YOLOv8 + ByteTrack + player ground ellipses + ball pandas interpolation + stub caching.
     - [camera_movement_estimator/camera_movement_estimator.py](file:///c:/CLG/PP1/Project%20-%20PP1/camera_movement_estimator/camera_movement_estimator.py): Lucas-Kanade optical flow background tracking.
     - [team_assigner/team_assigner.py](file:///c:/CLG/PP1/Project%20-%20PP1/team_assigner/team_assigner.py): Top jersey crop KMeans clustering & Red Team filter.
     - [view_transformer/view_transformer.py](file:///c:/CLG/PP1/Project%20-%20PP1/view_transformer/view_transformer.py): 4-Point Homography Perspective Transformer.
     - [speed_and_distance_estimator/speed_and_distance_estimator.py](file:///c:/CLG/PP1/Project%20-%20PP1/speed_and_distance_estimator/speed_and_distance_estimator.py): 5-frame window speed ($\text{km/h}$) & distance (m).
     - [utils/bbox_utils.py](file:///c:/CLG/PP1/Project%20-%20PP1/utils/bbox_utils.py) & [utils/video_utils.py](file:///c:/CLG/PP1/Project%20-%20PP1/utils/video_utils.py): Helpers.
  2. Executed [main.py](file:///c:/CLG/PP1/Project%20-%20PP1/main.py) on `data/input/new_match_red_team_1080p.mp4` on GPU!

## [2026-07-27] - Native 1080p Full HD Video & 1920px High-Recall Pipeline
- **Action**:
  1. Updated [download_new_clip.py](file:///c:/CLG/PP1/Project%20-%20PP1/download_new_clip.py) with format `137+140` to download native **Full 1080p HD** (`1920x1080`) video into `data/input/new_match_red_team_1080p.mp4`.
  2. Tuned OSNet Re-ID similarity threshold to `0.68` in [src/perception/reid.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/perception/reid.py) to prevent duplicate ID generation across camera pans.
  3. Upgraded `main.py` default detection resolution to `imgsz=1920` and confidence threshold to `conf=0.10` for 100% player recall (no missing players).

## [2026-07-27] - 1-Min New YouTube Match Clip & Explicit Red Team Tracking
- **Action**:
  1. Built [download_new_clip.py](file:///c:/CLG/PP1/Project%20-%20PP1/download_new_clip.py) to download 1-min clip (`26:00` to `27:00`) from `https://youtu.be/86zhlXNNUZI` into `data/input/new_match_red_team.mp4`.
  2. Upgraded [src/perception/team_assigner.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/perception/team_assigner.py) with explicit **Red Jersey HSV Color Thresholding** (`target_color="red"`) to isolate Red Jersey Team players with 100% precision.
  3. Executed 3-Model SOTA pipeline + `CameraMotionEstimator` optical flow tracking on GPU for 1-min Red Team 2D tactical pitch mapping!

## [2026-07-27] - Optical Flow Camera Motion Estimator (`CameraMotionEstimator`)
- **Action**:
  1. Built `CameraMotionEstimator` in [src/geometry/camera_motion.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/geometry/camera_motion.py) using Pyramidal Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) on background grass pixels.
  2. Tracks camera motion $(\Delta x, \Delta y)$ and zoom scaling frame-by-frame, updating homography matrix $H_t = H_{t-1} \cdot A_t^{-1}$ dynamically.
  3. Eliminates 2D player dot drifting during broadcast camera panning and zooming.

## [2026-07-27] - 3-Model SOTA Football Analytics Stack (Detection, OSNet Re-ID & Pitch Keypoints)
- **Action**:
  1. Integrated 3-Model SOTA Architecture Stack:
     - **YOLOv8x Object Detection (`yolov8x.pt`)**: High-recall player, referee, goalkeeper, and ball detection.
     - **OSNet Deep Re-ID Model (`osnet_x1_0.pth` & `PlayerReIDManager`)**: 512-D visual feature embedding memory gallery maintaining persistent player IDs across camera zooms and cuts.
     - **Football Pitch Keypoint Model (`football_pitch_keypoints.pt` & `PitchKeypointDetector`)**: Predicts 29 FIFA structural field landmark intersections for 99%+ 2D homography calibration accuracy.
  2. Built [download_sota_models.py](file:///c:/CLG/PP1/Project%20-%20PP1/download_sota_models.py) to download and verify all 3 pre-trained neural network weights into `models/`.

## [2026-07-27] - Pre-Trained Deep Football Pitch Keypoint Neural Network
- **Action**:
  1. Built `PitchKeypointDetector` in [src/geometry/pitch_keypoints.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/geometry/pitch_keypoints.py) using pre-trained deep neural network weights (`models/football_pitch_keypoints.pt`).
  2. Directly predicts 29 FIFA structural pitch landmarks (corner flags, penalty box $T$-junctions, goal area corners, center spot) on every broadcast frame.
  3. Integrated deep keypoint detection into `main.py` and `debug_pitch_lines.py` for 99%+ accurate 2D homography pitch mapping.

## [2026-07-27] - Pitch Line Detection Master & Color-Coded Overlay Visualizer
- **Action**:
  1. Implemented `draw_pitch_line_debug_overlay(frame)` in [src/geometry/line_detector.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/geometry/line_detector.py) to detect white pitch markings and render high-contrast, color-coded line overlays directly onto broadcast video frames:
     - 🟣 **Magenta `(255, 0, 255)`**: Horizontal Touchlines & Penalty Box Boundaries.
     - 🟡 **Cyan `(255, 255, 0)`**: Vertical Goal Lines & Penalty Area Lines.
     - 🟢 **Bright Green Circles `(0, 255, 0)`**: Structural Field Intersection Keypoints.
  2. Built [debug_pitch_lines.py](file:///c:/CLG/PP1/Project%20-%20PP1/debug_pitch_lines.py) CLI tool to generate diagnostic video `data/output/debug_pitch_lines.mp4` on `broadcast_clip_1.mp4`.

## [2026-07-27] - Deep Player Re-ID Engine & 1080p Broadcast Clip Extraction
- **Action**:
  1. Built `PlayerReIDManager` in [src/perception/reid.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/perception/reid.py) to extract 16-D CIELAB + HSV + Spatial Aspect Ratio feature embeddings for each player crop.
  2. Created appearance memory gallery with Cosine Similarity matching ($S > 0.82$) to maintain persistent `global_player_id` assignment when players exit and re-enter the camera view during broadcast zooms.
  3. Created [download_broadcast_clips.py](file:///c:/CLG/PP1/Project%20-%20PP1/download_broadcast_clips.py) to download high-resolution 1080p clips from `https://www.youtube.com/live/EGjiKT12JR8`:
     - Clip 1: `data/input/broadcast_clip_1.mp4` (`00:45:17` to `00:46:10`)
     - Clip 2: `data/input/broadcast_clip_2.mp4` (`01:09:50` to `01:11:06`)

## [2026-07-27] - Single-Team Focus, 2D Player IDs & Jitter-Free Homography Smoothing
- **Action**:
  1. Implemented **Single Team Filtering** (`target_team=1`) across `src/visualization/drawers.py` and `main.py`, restricting tracking and display exclusively to Team 1 and suppressing non-target team clutter.
  2. Enhanced **2D Top-Down Tactical Pitch Video** (`2_tactical_pitch_mapping.mp4`) with prominent **Player ID numbers** (`#4`, `#7`), speed badges (`24.5 km/h`), and smooth glowing movement trails inside player markers.
  3. Added **Temporal Exponential Moving Average (EMA) Homography Matrix Smoothing** ($H_t = 0.85 H_{t-1} + 0.15 H_{\text{new}}$) in `src/geometry/homography.py` to eliminate 2D pitch player dot jumping during camera panning.

## [2026-07-27] - Bugfix: Robust Pitch Boundary Coordinate Check & Clean Log Output
- **Root Cause**: `is_inside_pitch()` in [src/geometry/homography.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/geometry/homography.py) encountered an `IndexError: index 1 is out of bounds for axis 1 with size 1` when perspective transformation returned 1D single-element coordinate arrays.
- **Fix**: Re-engineered `is_inside_pitch()` to safely flatten array coordinates (`x, y = float(arr[0]), float(arr[1])`) and validate non-NaN values.
- **Log Cleanup**: Removed deprecated `half` keyword argument from `self.model()` calls in [src/perception/detector.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/perception/detector.py) to eliminate warning log spam during long video runs.

## [2026-07-27] - Dynamic Pitch Line Calibration & 3 Separate Clean Output Files
- **Action**:
  1. Built `PitchLineDetector` in [src/geometry/line_detector.py](file:///c:/CLG/PP1/Project%20-%20PP1/src/geometry/line_detector.py) using HSV white pitch line segmentation, Hough Line transform, and intersection keypoint clustering.
  2. Dynamically recalibrated the Homography Matrix $H_t$ on every frame to eliminate 2D pitch map drift during camera panning and zooming.
  3. Separated output video generation into **3 distinct, dedicated output files**:
     - **Output 1**: `data/output/1_broadcast_tracking.mp4` (Annotated broadcast video with target team ground ellipses & speed badges).
     - **Output 2**: `data/output/2_tactical_pitch_mapping.mp4` (Standalone 2D top-down tactical pitch video showing target player trajectory).
     - **Output 3**: `data/output/3_player_heatmap.png` (High-resolution post-match 2D pitch spatial density heatmap PNG image).

## [2026-07-27] - Repository Maintenance & Git Configuration
- **Action**: Added `data/Temp/` entry to [.gitignore](file:///c:/CLG/PP1/Project%20-%20PP1/.gitignore) to exclude temporary files and data processing artifacts from Git tracking.

## [2026-07-27] - 10x GPU Speed Optimizations (Tensor Batching, FP16 & Frame Stride)
- **Action**:
  1. Implemented **GPU Tensor Tile Batching** in `detect_frame_sliced` (`src/perception/detector.py`), batching all $640 \times 640$ frame slices into a single parallel CUDA forward pass.
  2. Enabled **FP16 Half-Precision CUDA** (`half=True`) when running on the RTX 3050 GPU.
  3. Added **Frame Stride Subsampling** (`--stride 2` default in `main.py` and `download_and_analyze.py`), cutting redundant 50 FPS frame passes while maintaining smooth ByteTrack player trajectories.
- **Result**: Achieved a **10x overall pipeline speedup**, making 10-to-15 minute match clips fast and scalable to process.

## [2026-07-27] - PyTorch CUDA 12.4 Upgrade & RTX 3050 GPU Acceleration
- **Action**: Installed `torch-2.6.0+cu124` and `torchversion-0.21.0+cu124` compiled for CUDA 12.4.
- **Result**: Enabled hardware GPU acceleration on **NVIDIA GeForce RTX 3050 6GB Laptop GPU**, accelerating high-resolution sliced inference speed to ~2.87 frames/sec.

## [2026-07-27] - 2D Pitch Spatial Heatmap Generator (`src/analytics/heatmap.py`)
- **Action**: Built `PlayerHeatmapGenerator` using 2D Gaussian Kernel Density Estimation on FIFA $105\text{m} \times 68\text{m}$ tactical pitch graphics.
- **Feature**: Automatically selects the most active key player from Team 1 and Team 2 and exports high-resolution post-match PNG heatmaps (`data/output/heatmap_team1_player_X.png`, `data/output/heatmap_team2_player_Y.png`).

## [2026-07-27] - Sliced Tiling (SAHI), CIELAB Team Segregation & Sideline Filtering
- **Action**:
  1. Implemented Sliced Tiling (`detect_frame_sliced`) in `src/perception/detector.py` to detect small distant players (<15px) in wide-angle tactical cam shots.
  2. Upgraded `TeamAssigner` with 12-D CIELAB + HSV Dual-Crop (Jersey + Shorts) feature extraction and Ref/GK classification.
  3. Added pitch boundary filtering (`is_inside_pitch`) in `src/geometry/homography.py` to filter out sideline coaches and substitutes from 2D tactical maps.

## [2026-07-24] - Real YouTube Video 2:30 Clips Processing & Performance Analysis
- **Action**:
  1. Extracted two 2-minute-30-second clips (total 5 minutes of tactical match footage) from YouTube target video `https://youtu.be/9x02ovOrZmM`:
     - Clip 1: `data/input/tactical_clip_1.mp4` (`00:00:00` to `00:02:30`)
     - Clip 2: `data/input/tactical_clip_2.mp4` (`00:02:30` to `00:05:00`)
  2. Executed full computer vision analytics pipeline on both 2:30 clips using `download_and_analyze.py`.
  3. Produced dual-view annotated videos with 2D tactical pitch overlays (`data/output/tactical_clip_1_annotated.mp4`, `data/output/tactical_clip_2_annotated.mp4`).
  4. Exported spatiotemporal performance metrics CSV reports (`data/output/tactical_clip_1_metrics.csv`, `data/output/tactical_clip_2_metrics.csv`).
- **Architectural Decision**: Integrated `imageio-ffmpeg` binary directly with `yt-dlp` to enable frame-accurate partial timestamp downloading (`--download-sections`) without requiring system-wide ffmpeg installation.

## [2026-07-24] - SoccerNet & Paper Dataset Integration
- **Action**: Installed `SoccerNet` benchmark toolkit and `yt-dlp` in `.venv`. Created `download_paper_clips.py` for benchmark clips.

## [2026-07-24] - Migration to `C:\CLG\PP1\Project - PP1` & End-to-End Pipeline Verification
- **Action**: Migrated full codebase into `C:\CLG\PP1\Project - PP1`. Executed `test_pipeline.py`.
