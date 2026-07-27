# Development Log

## [2026-07-27] - PyTorch CUDA 12.4 Upgrade & RTX 3050 GPU Acceleration
- **Action**: Installed `torch-2.6.0+cu124` and `torchvision-0.21.0+cu124` compiled for CUDA 12.4.
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
