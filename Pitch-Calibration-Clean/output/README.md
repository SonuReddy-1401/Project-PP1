# Pitch-Calibration-Clean / Output

This directory stores exported camera calibration parameters and visual verification plots.

## Expected Content
- `predictions/`: Per-frame JSON files (`camera_*.json`) containing pan, tilt, roll, and focal length calibration parameters.
- `visual_verification/`: Annotated camera projection verification images (`*_annotated.jpg`).

## Generation
- Populated by running `visualize_calibration.py` or inference export scripts.
- Generated output files are excluded from Git to prevent repository bloat.
