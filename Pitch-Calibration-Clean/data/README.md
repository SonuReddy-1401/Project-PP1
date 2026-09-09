# Pitch-Calibration-Clean / Data

This directory stores input test frames and video datasets used for camera calibration testing.

## Expected Content
- Frame image files (`frame_*.jpg`) extracted from match clips.
- Subfolders corresponding to test broadcast video clips (`clip1/`, `clip2/`).

## Setup Instructions
- Frame images are automatically placed here during pre-processing using `prepare_full_frames_clip2.py` or manual video frame extraction.
- Heavy dataset binaries (`*.jpg`, `*.mp4`) are excluded from Git version control.
