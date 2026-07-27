import cv2
import os
from tqdm import tqdm

def read_video(video_path: str):
    """
    Reads all frames from a video file into a list of BGR numpy image frames with progress bar.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    pbar = tqdm(total=total_frames, desc="[0/5] Loading Video Frames Into Memory")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        pbar.update(1)
    cap.release()
    pbar.close()
    print(f"[INFO] Loaded {len(frames)} total frames from {video_path}")
    return frames

def save_video(output_video_frames: list, output_video_path: str, fps: float = 25.0):
    """
    Saves a list of BGR image frames to an output video file using OpenCV VideoWriter with progress bar.
    """
    if not output_video_frames:
        print("[WARNING] No frames to save!")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    height, width, _ = output_video_frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    base_name = os.path.basename(output_video_path)
    for frame in tqdm(output_video_frames, desc=f"[EXPORT] Streaming Video File: {base_name}"):
        out.write(frame)
    out.release()
    print(f"[SUCCESS] Saved output video ({len(output_video_frames)} frames) to: {output_video_path}")
