import os
import cv2
import numpy as np

def inspect_video_frames(video_path: str = "data/input/new_match_red_team_1080p.mp4", out_dir: str = "data/Temp"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    frames_to_save = [0, 250, 500, 750, 1000]
    curr = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if curr in frames_to_save:
            out_img = os.path.join(out_dir, f"frame_{curr}.jpg")
            cv2.imwrite(out_img, frame)
            print(f"[SAVED] {out_img} ({frame.shape[1]}x{frame.shape[0]})")
            saved += 1

        curr += 1
        if curr > 1000:
            break

    cap.release()

if __name__ == "__main__":
    inspect_video_frames()
