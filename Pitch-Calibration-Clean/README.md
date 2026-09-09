# Pitch Calibration Clean

This rebuild separates calibration from rendering.  Stage D reads only the
camera JSON files emitted by the copied SoccerNet baseline; it does not perform
live homography estimation.

Run from this directory with a working Python environment:

```powershell
python prepare_full_frames.py
cd sn-calibration
python src/detect_extremities.py -s ..\data\custom_dataset -p ..\output\predictions --split test
python src/baseline_cameras.py -s ..\data\custom_dataset -p ..\output\predictions --split test
cd ..
python render_calibrated_video.py
```

`prepare_full_frames.py` records the actual source FPS in `data/video_info.json`.
The renderer reads it automatically, or accepts `--fps` to override it.
