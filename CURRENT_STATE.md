# Current Project State

## Status Overview
- **Active Project Root**: `C:\CLG\PP1\Project - PP1`
- **Phase**: Phase 1 - Recreating `abdullahtarek/football_analysis` Pipeline Architecture ACTIVE
- **Hardware Acceleration**: **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (PyTorch 2.6.0+cu124 CUDA active)
- **Active Video**: `data/input/new_match_red_team_1080p.mp4` (Native 1920x1080 Full HD clip from `https://youtu.be/86zhlXNNUZI`)
- **Active Architecture Modules**:
  - **`trackers/tracker.py`**: YOLOv8x + ByteTrack tracking, player ground ellipses, track ID badges, ball triangle pointers, pandas ball interpolation, stub pickle caching (`stubs_tracks.pkl`).
  - **`camera_movement_estimator/`**: Optical Flow Lucas-Kanade background tracking (`stubs_camera_movement.pkl`).
  - **`team_assigner/`**: Top jersey crop color segmentation & Red Jersey HSV filtering.
  - **`view_transformer/`**: 4-Point Homography Perspective Transformation matrix (Pixels -> 2D Pitch Canvas Meters).
  - **`speed_and_distance_estimator/`**: 5-frame window speed ($\text{km/h}$) & distance (m) kinematic engine.
- **Overall System Readiness**: 100%

## Output Files & Active Video Clips
- **Target YouTube Video**: `https://youtu.be/86zhlXNNUZI` (Timestamp `26:00` to `27:00`)
- **Native 1080p HD Input Clip**: `data/input/new_match_red_team_1080p.mp4` (1920x1080 Full HD, 60s)
- **Output 1 (Broadcast Red Team Tracking)**: `data/output/1_broadcast_tracking.mp4`
- **Output 2 (2D Red Team Tactical Pitch Video)**: `data/output/2_tactical_pitch_mapping.mp4`

## Architecture Modules & Scripts
- [x] `trackers/tracker.py` (**NEW**: YOLOv8 + ByteTrack + Ellipses + Ball Interpolation + Pickle Caching)
- [x] `camera_movement_estimator/camera_movement_estimator.py` (**NEW**: Optical Flow Lucas-Kanade Camera Motion Estimator)
- [x] `team_assigner/team_assigner.py` (**NEW**: Top Jersey Crop KMeans Color Clustering + Red Jersey Filter)
- [x] `view_transformer/view_transformer.py` (**NEW**: 4-Point Homography Perspective Transformer)
- [x] `speed_and_distance_estimator/speed_and_distance_estimator.py` (**NEW**: Kinematic Speed km/h & Distance m Estimator)
- [x] `utils/bbox_utils.py` (**NEW**: Bounding Box Center, Width, Foot Coordinate Helpers)
- [x] `utils/video_utils.py` (**NEW**: OpenCV Video Frame Reader & Writer)
- [x] `download_new_clip.py` (Downloader for Native 1080p Full HD timestamp clip `26:00` to `27:00`)
- [x] `download_sota_models.py` (Downloader for all 3 SOTA model weights into `models/`)
- [x] `main.py` (**NEW**: Clean Abdullah Tarek Pipeline Orchestration Entry Point)

## Environment Details
- **Location**: `C:\CLG\PP1\Project - PP1`
- **Python**: 3.13.12 (`.venv` active)
- **PyTorch**: `2.6.0+cu124` (CUDA 12.4 enabled)
- **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
