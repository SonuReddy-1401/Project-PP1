# Next Tasks

## Completed Tasks
- [x] Migrate full project codebase to `C:\CLG\PP1\Project - PP1`.
- [x] Build isolated `.venv` environment and install all dependencies.
- [x] Upgrade PyTorch to `2.6.0+cu124` to enable hardware acceleration on NVIDIA GeForce RTX 3050 6GB GPU.
- [x] Implement modular pipeline architecture (`src/utils`, `src/perception`, `src/geometry`, `src/analytics`, `src/visualization`).
- [x] Implement Sliced Tiling (`SAHI`) in `detector.py` for ultra-high-recall small player detection in wide tactical shots.
- [x] Upgrade `team_assigner.py` with 12-D CIELAB + HSV Dual-Crop color feature extraction and Ref/GK classification.
- [x] Add pitch boundary sideline filtering (`is_inside_pitch`) in `homography.py` to suppress coaches and sub bench from 2D maps.
- [x] Implement `PlayerHeatmapGenerator` in `src/analytics/heatmap.py` to auto-export post-match 2D pitch spatial density heatmaps for key players.
- [x] Implement `main.py` CLI pipeline entry point and `test_pipeline.py` synthetic test harness.

## Upcoming Performance & Feature Roadmap Tasks
- [ ] Implement Frame Stride sampling (`--stride 2` / `stride 3`) for 50%–66% computation time reduction.
- [ ] Implement GPU Tensor Batching (`batch_size=6`) to run all frame tiles in a single CUDA forward pass.
- [ ] Enable FP16 Half-Precision CUDA (`half=True`) for 2x RTX 3050 Tensor Core throughput.
- [ ] Phase 2: Add automatic pitch keypoint detection for dynamic line-based homography recalibration.
- [ ] Phase 2: Add team ball possession tracker and pass counter.
- [ ] Phase 3: Add broadcast replay and commercial break scene-cut classifier.
