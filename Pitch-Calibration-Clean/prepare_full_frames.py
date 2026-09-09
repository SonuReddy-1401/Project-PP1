"""
Extract ALL frames (native fps) from input video for full video rendering.
Phase 1 Clean Rebuild - Stage A
"""
import os
import json
import cv2

def extract_all_frames(video_path: str, output_dataset_dir: str, clip_id: str = "clip1"):
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
    print(f"[INFO] Extracting ALL {total_frames} frames at native {video_fps:.2f} fps")

    saved_frames = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = f"frame_{frame_count:05d}.jpg"
        relative_path = os.path.join(clip_id, frame_filename).replace("\\", "/")
        cv2.imwrite(os.path.join(clip_dir, frame_filename), frame)
        saved_frames[relative_path] = {}

    cap.release()

    manifest_path = os.path.join(output_dataset_dir, "test", "per_match_info.json")
    with open(manifest_path, "w") as f:
        json.dump({clip_id: saved_frames}, f, indent=4)

    print(f"[SUCCESS] Extracted {frame_count} frames to: {clip_dir}")
    print(f"[SUCCESS] Video fps for later reconstruction: {video_fps}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_VIDEO = r"C:\CLG\PP1\EXP\Pitch-Segmentation\data\input_video.mp4"
    OUTPUT_DATASET = os.path.join(SCRIPT_DIR, "data", "custom_dataset")
    extract_all_frames(INPUT_VIDEO, OUTPUT_DATASET, clip_id="clip1")
