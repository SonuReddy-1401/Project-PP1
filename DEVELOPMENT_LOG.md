# Development Log

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
