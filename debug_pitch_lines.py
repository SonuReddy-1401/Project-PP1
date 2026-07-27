import os
import argparse
import cv2
from tqdm import tqdm
from src.utils.video_utils import get_video_properties, VideoWriterStream
from src.geometry.line_detector import PitchLineDetector
from src.geometry.pitch_keypoints import PitchKeypointDetector

def generate_pitch_line_debug_video(input_video: str, output_video: str, max_frames: int = None, use_deep: bool = True):
    """
    Executes PitchLineDetector or PitchKeypointDetector on input video and exports debug video.
    """
    print("=" * 70)
    print("      DEEP PITCH KEYPOINT & LINE DETECTION DEBUGGER      ")
    print("=" * 70)
    
    props = get_video_properties(input_video)
    print(f"[INFO] Input Video: {input_video}")
    print(f"       Resolution: {props['width']}x{props['height']} | FPS: {props['fps']} | Total Frames: {props['total_frames']}")
    print(f"[INFO] Output Debug Video: {output_video}")

    line_detector = PitchLineDetector()
    deep_detector = PitchKeypointDetector() if use_deep else None

    cap = cv2.VideoCapture(input_video)
    writer = None
    frame_idx = 0

    total = min(props['total_frames'], max_frames) if max_frames else props['total_frames']
    pbar = tqdm(total=total, desc="[DETECTING PITCH KEYPOINTS]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Generate Pitch Line & Keypoint Debug Overlay
        debug_frame = line_detector.draw_pitch_line_debug_overlay(frame)
        if deep_detector is not None:
            debug_frame = deep_detector.draw_keypoint_overlay(debug_frame)

        if writer is None:
            h, w, _ = debug_frame.shape
            writer = VideoWriterStream(output_video, fps=props['fps'], frame_size=(w, h))

        writer.write(debug_frame)
        frame_idx += 1
        pbar.update(1)

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    pbar.close()

    print("\n[SUCCESS] Pitch keypoint debug video successfully generated!")
    print(f"Debug Output Video: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch Line & Keypoint Debugger Generator")
    parser.add_argument("--input", type=str, default="data/input/broadcast_clip_1.mp4", help="Path to input video clip")
    parser.add_argument("--output", type=str, default="data/output/debug_pitch_lines.mp4", help="Path to output debug video")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames limit")
    args = parser.parse_args()

    generate_pitch_line_debug_video(args.input, args.output, max_frames=args.max_frames)
