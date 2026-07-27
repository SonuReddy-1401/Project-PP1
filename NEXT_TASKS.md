# Next Tasks
# Next Tasks & Roadmap

## 📌 Active Branch: `ABDULLAH`

### Phase 1: Git Branch `ABDULLAH` Core Files Staged & Committed
- [x] Recreate 5-module Abdullah Tarek pipeline (`trackers/`, `camera_movement_estimator/`, `team_assigner/`, `view_transformer/`, `speed_and_distance_estimator/`).
- [x] Configure `.gitignore` to exclude large model binaries (`models/`) and stub pickles (`*.pkl`).
- [x] Commit core pipeline code to branch `ABDULLAH`.

### Phase 2: Next Quality Enhancements (To reach 100% output)
- [ ] **Dynamic Ball Control & Possession Metrics**: Calculate live percentage ball possession per team based on ball proximity.
- [ ] **Multi-Camera Perspective Alignment**: Improve homography for camera zoom-ins and panning.
- [ ] **Player Pass & Trajectory Network**: Map passes between team members in 2D space.

## Completed Tasks
- [x] Migrate full project codebase to `C:\CLG\PP1\Project - PP1`.
- [x] Upgrade PyTorch to `2.6.0+cu124` to enable hardware acceleration on NVIDIA GeForce RTX 3050 6GB GPU.
- [x] Implement Sliced Tiling (`SAHI`) in `detector.py` for ultra-high-recall small player detection in wide tactical shots.
- [x] Upgrade `team_assigner.py` with 12-D CIELAB + HSV Dual-Crop color feature extraction and Ref/GK classification.
- [x] Implement `PlayerReIDManager` in `src/perception/reid.py` for 16-D appearance embedding feature extraction and gallery matching across camera zooms.
- [x] Implement `download_broadcast_clips.py` for 1080p YouTube timestamp section extraction.
- [x] Add dynamic pitch line calibration (`PitchLineDetector`) and EMA matrix smoothing in `homography.py` to eliminate 2D dot jumping.
- [x] Implement `PlayerHeatmapGenerator` in `src/analytics/heatmap.py` to auto-export post-match 2D pitch spatial density heatmaps for key players.
- [x] Stream 3 separate clean output files (`1_broadcast_tracking.mp4`, `2_tactical_pitch_mapping.mp4`, `3_player_heatmap.png`).

## Upcoming Roadmap Tasks
- [ ] Phase 2: Add team ball possession tracker and pass counter.
- [ ] Phase 3: Add broadcast replay and commercial break scene-cut classifier.
