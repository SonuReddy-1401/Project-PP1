"""
Prepare Test Frames & SoccerNet Data Hierarchy
Phase 1 - SoccerNet Camera Calibration Baseline

Extracts frames at 1 FPS from input video clips and creates the per_match_info.json
manifest file expected by detect_extremities.py and baseline_cameras.py.

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation-2
"""

import os
import json
import cv2

def extract_frames_and_build_manifest(
    video_path: str,
    output_dataset_dir: str,
    clip_id: str = "clip1",
    sample_fps: float = 1.0
):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    clip_dir = os.path.join(output_dataset_dir, "test", clip_id)
    os.makedirs(clip_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))

    print(f"[INFO] Processing Video: {video_path}")
    print(f"[INFO] Video FPS: {video_fps:.2f} | Extracting at: {sample_fps} FPS (Every {frame_interval} frames)")

    saved_frames = {}
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            saved_count += 1
            frame_filename = f"frame_{saved_count:05d}.jpg"
            relative_path = os.path.join(clip_id, frame_filename).replace("\\", "/")
            full_frame_path = os.path.join(clip_dir, frame_filename)

            cv2.imwrite(full_frame_path, frame)
            saved_frames[relative_path] = {}

        frame_count += 1

    cap.release()

    # Create per_match_info.json inside test directory
    manifest_path = os.path.join(output_dataset_dir, "test", "per_match_info.json")
    manifest = {clip_id: saved_frames}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"[SUCCESS] Extracted {saved_count} frames to: {clip_dir}")
    print(f"[SUCCESS] Created SoccerNet manifest at: {manifest_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_VIDEO = r"C:\CLG\PP1\EXP\Pitch-Segmentation\data\input_video.mp4"
    OUTPUT_DATASET = os.path.join(SCRIPT_DIR, "data", "custom_dataset")

    extract_frames_and_build_manifest(
        video_path=INPUT_VIDEO,
        output_dataset_dir=OUTPUT_DATASET,
        clip_id="clip1",
        sample_fps=1.0
    )
