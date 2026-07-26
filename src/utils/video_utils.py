import cv2
import os
from typing import List, Generator, Tuple, Dict, Any

def get_video_properties(video_path: str) -> Dict[str, Any]:
    """
    Extracts metadata from a video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30.0,
        "total_frames": total_frames
    }

def read_video_frames(video_path: str, max_frames: int = None) -> List[Any]:
    """
    Reads all frames from a video into memory (for short clips).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return frames

def generate_video_frames(video_path: str) -> Generator[Tuple[int, Any], None, None]:
    """
    Frame generator to stream video frame by frame (memory efficient).
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1
    cap.release()

class VideoWriterStream:
    """
    Streaming video writer that writes frames directly to disk with O(1) memory usage.
    """
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.output_path = output_path

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        print(f"[INFO] Video successfully saved to: {self.output_path}")

def save_video_frames(output_path: str, frames: List[Any], fps: float = 30.0):
    """
    Saves a list of frames as an output MP4 video.
    """
    if not frames:
        raise ValueError("No frames provided to save.")
        
    height, width, _ = frames[0].shape
    writer = VideoWriterStream(output_path, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
