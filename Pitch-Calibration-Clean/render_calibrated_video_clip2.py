"""
Render full video for clip2 from pre-computed verified camera calibration JSON files.
Phase 1 Clean Rebuild - Stage 3b (clip2)
"""
import os
import sys
import json
import glob
import re
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SN_CALIB_DIR = os.path.join(SCRIPT_DIR, "sn-calibration")
sys.path.append(SN_CALIB_DIR)

from src.camera import Camera
from src.soccerpitch import SoccerPitch

FRAMES_DIR = os.path.join(SCRIPT_DIR, "data", "custom_dataset", "test", "clip2")
CAMERA_DIR = os.path.join(SCRIPT_DIR, "output", "predictions", "test", "camera_clip2")
OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "output", "calibrated_video_clip2.mp4")
CALIB_RESOLUTION_W, CALIB_RESOLUTION_H = 960, 540

def frame_number_from_path(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def main():
    pitch = SoccerPitch()

    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")),
                          key=frame_number_from_path)
    if not frame_files:
        print(f"[ERROR] No frames found in {FRAMES_DIR}")
        return

    first_frame = cv2.imread(frame_files[0])
    h_orig, w_orig = first_frame.shape[:2]
    scale_factor = w_orig / CALIB_RESOLUTION_W

    # Detect video FPS from first frame's original video if available, default to 25.0
    video_fps = 25.0
    video_path = os.path.join(SCRIPT_DIR, "data", "broadcast_clip_1.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()

    print(f"[INFO] Rendering clip2 video ({len(frame_files)} frames) at {video_fps:.2f} FPS")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                              video_fps, (w_orig, h_orig))

    last_known_params = None
    solved_count = 0
    fallback_count = 0
    blank_count = 0

    for i, frame_path in enumerate(frame_files):
        frame_num = frame_number_from_path(frame_path)
        cam_json_path = os.path.join(CAMERA_DIR, f"frame_{frame_num:05d}.json")

        image = cv2.imread(frame_path)

        cam_params = None
        if os.path.exists(cam_json_path):
            with open(cam_json_path, "r") as f:
                cam_params = json.load(f)
            if cam_params:
                last_known_params = cam_params
                solved_count += 1
            else:
                cam_params = last_known_params
                fallback_count += 1
        else:
            cam_params = last_known_params
            fallback_count += 1

        if cam_params:
            camera = Camera(CALIB_RESOLUTION_W, CALIB_RESOLUTION_H)
            camera.from_json_parameters(cam_params)
            if (w_orig, h_orig) != (CALIB_RESOLUTION_W, CALIB_RESOLUTION_H):
                camera.scale_resolution(scale_factor)
            annotated = camera.draw_colorful_pitch(image, pitch.palette)
        else:
            annotated = image
            blank_count += 1

        writer.write(annotated)

        if (i + 1) % 100 == 0 or (i + 1) == len(frame_files):
            print(f"[PROGRESS] {i+1}/{len(frame_files)} | "
                  f"Solved: {solved_count} | Fallback: {fallback_count} | Blank: {blank_count}")

    writer.release()
    print(f"\n[SUCCESS] Clip2 video saved to {OUTPUT_VIDEO}")
    print(f"[SUMMARY] Solved: {solved_count} | Fallback (used last known): {fallback_count} | Blank (no calibration yet): {blank_count}")

if __name__ == "__main__":
    main()
