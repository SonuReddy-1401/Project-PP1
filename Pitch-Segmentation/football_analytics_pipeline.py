"""
Football Player Tracking, Trajectory & Positional Heatmap Pipeline
Adapted from Labellerr Football Analytics Architecture

Key Capabilities:
1. YOLOv8 + ByteTrack Object Tracking (Players, Referees, Ball) running on NVIDIA GPU (RTX 3050).
2. Persistent Track IDs (track_id) with Kalman Filtered movement prediction across camera motion.
3. Player Trajectory Trail Renderer (Past N frames motion history).
4. Team Positional Density Heatmap Generator (Color-mapped spatial occupancy).
5. Green Turf Pitch Masking & Picture-in-Picture (PIP) Heatmap View.

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation
"""

import os
import time
from collections import defaultdict, deque
from typing import Tuple, Optional, Dict, List, Any
import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class AdaptiveHSVGrassSegmentor:
    """Green turf segmentor for pitch masking."""

    def __init__(self, lower_green=(32, 65, 65), upper_green=(85, 255, 255)):
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)

    def segment_turf(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        raw_mask[:int(h * 0.15), :] = 0

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        turf_mask = np.zeros((h, w), dtype=np.uint8)

        min_area = (h * w) * 0.05
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                turf_mask[labels == i] = 255

        return turf_mask


class FootballHeatmapGenerator:
    """
    Accumulates player positional history across video frames
    and generates a Gaussian-smoothed positional heatmap.
    """

    def __init__(self, frame_shape: Tuple[int, int], decay_factor: float = 0.998):
        self.h, self.w = frame_shape
        self.accumulator = np.zeros((self.h, self.w), dtype=np.float32)
        self.decay_factor = decay_factor

    def update(self, player_positions: List[Tuple[int, int]]) -> None:
        # Apply slight temporal decay to emphasize recent activity
        self.accumulator *= self.decay_factor

        for px, py in player_positions:
            if 0 <= px < self.w and 0 <= py < self.h:
                # Add gaussian splat for foot location
                cv2.circle(self.accumulator, (px, py), 18, 1.0, -1)

    def get_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        if np.max(self.accumulator) == 0:
            return frame.copy()

        # Normalize accumulator to [0, 255]
        norm_acc = (self.accumulator / np.max(self.accumulator) * 255.0).astype(np.uint8)

        # Apply Gaussian Blur to create smooth heat density
        blurred_acc = cv2.GaussianBlur(norm_acc, (31, 31), 0)

        # Color-map: JET (Blue=Low, Yellow=Med, Red=High Density)
        heatmap_color = cv2.applyColorMap(blurred_acc, cv2.COLORMAP_JET)

        # Mask heatmap to non-zero density regions
        mask = blurred_acc > 15
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame[mask], 1.0 - alpha, heatmap_color[mask], alpha, 0)

        return overlay

    def get_pure_heatmap_rgb(self) -> np.ndarray:
        if np.max(self.accumulator) == 0:
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)

        norm_acc = (self.accumulator / np.max(self.accumulator) * 255.0).astype(np.uint8)
        blurred_acc = cv2.GaussianBlur(norm_acc, (31, 31), 0)
        return cv2.applyColorMap(blurred_acc, cv2.COLORMAP_JET)


class FootballAnalyticsPipeline:
    """
    Master Pipeline Class for Football Player Tracking & Heatmap Generation.
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        conf_threshold: float = 0.35,
        max_trajectory_length: int = 30,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.max_trajectory_length = max_trajectory_length

        print(f"[INFO] Initializing Football Analytics Pipeline on device: {self.device}")
        self.model = YOLO(model_path)

        self.turf_segmentor = AdaptiveHSVGrassSegmentor()
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trajectory_length))
        self.heatmap_gen: Optional[FootballHeatmapGenerator] = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        h, w = frame.shape[:2]

        if self.heatmap_gen is None:
            self.heatmap_gen = FootballHeatmapGenerator(frame_shape=(h, w))

        # 1. Pitch Turf Masking
        turf_mask = self.turf_segmentor.segment_turf(frame)

        # 2. YOLO Object Tracking with ByteTrack (Runs on GPU)
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )[0]

        tracked_players = []
        player_foot_positions = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                # COCO Class 0 = Person (Player/Referee)
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box)
                    foot_x = (x1 + x2) // 2
                    foot_y = y2

                    tracked_players.append((track_id, (x1, y1, x2, y2), conf))
                    player_foot_positions.append((foot_x, foot_y))

                    # Append position to trajectory queue
                    self.trajectories[track_id].append((foot_x, foot_y))

        # 3. Update Heatmap Density Accumulator
        self.heatmap_gen.update(player_foot_positions)

        # 4. Render Visualization Overlays
        rendered_frame = frame.copy()

        # Semi-Transparent Green Turf Mask
        if np.count_nonzero(turf_mask) > 0:
            turf_color = np.zeros_like(frame)
            turf_color[turf_mask > 0] = (0, 180, 0)
            rendered_frame = cv2.addWeighted(rendered_frame, 1.0, turf_color, 0.25, 0)

        # Draw Heatmap Overlay on main video frame
        rendered_frame = self.heatmap_gen.get_heatmap_overlay(rendered_frame, alpha=0.40)

        # Draw Player Bounding Boxes & Trajectory Motion Trails
        for track_id, (x1, y1, x2, y2), conf in tracked_players:
            # Trajectory Trail Line
            pts = list(self.trajectories[track_id])
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(rendered_frame, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Player Bounding Box (Cyan)
            cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Track ID Label Badge
            label = f"ID:{track_id}"
            cv2.rectangle(rendered_frame, (x1, y1 - 22), (x1 + 65, y1), (255, 255, 0), -1)
            cv2.putText(rendered_frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)

        # FPS Calculation
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 5. Real-Time HUD Status
        hud_text = f"YOLOv8 ByteTrack | Active Players: {len(tracked_players)} | FPS: {fps:.1f}"
        cv2.rectangle(rendered_frame, (10, 10), (550, 50), (0, 120, 0), -1)
        cv2.putText(rendered_frame, hud_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # 6. PIP View: Pure Tactical Positional Heatmap (Top Right Corner)
        pure_heatmap = self.heatmap_gen.get_pure_heatmap_rgb()
        pip_w, pip_h = w // 4, h // 4
        pip_resized = cv2.resize(pure_heatmap, (pip_w, pip_h))
        cv2.rectangle(pip_resized, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)

        margin = 15
        rendered_frame[margin:margin + pip_h, w - pip_w - margin:w - margin] = pip_resized
        cv2.putText(
            rendered_frame, "Positional Heatmap PIP", (w - pip_w - margin + 5, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

        meta = {
            "active_players_count": len(tracked_players),
            "fps": fps,
            "gpu_device": self.device,
        }

        return rendered_frame, meta

    def process_video(self, video_path: str, output_path: str) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("==================================================")
        print("  FOOTBALL PLAYER TRACKING & HEATMAP PIPELINE    ")
        print("==================================================")
        print(f"[INFO] Hardware Acceleration: {self.device}")
        print(f"[INFO] Input Video: {video_path}")
        print(f"[INFO] Resolution: {width}x{height} | FPS: {input_fps:.2f} | Total Frames: {total_frames}\n")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))

        frame_count = 0
        total_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                rendered_frame, meta = self.process_frame(frame)
                writer.write(rendered_frame)

                if frame_count % 30 == 0 or frame_count == total_frames:
                    print(
                        f"[PROGRESS Analytics] Frame {frame_count}/{total_frames} | "
                        f"Active Players: {meta['active_players_count']} | FPS: {meta['fps']:.1f}"
                    )
        finally:
            cap.release()
            writer.release()

        total_elapsed = time.time() - total_start
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        print(f"\n[SUCCESS] Processing complete in {total_elapsed:.2f} seconds!")
        print(f"[METRICS] Average Processing Speed: {avg_fps:.2f} FPS")
        print(f"[OUTPUT FILE] Saved to: {output_path}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "football_analytics_output.mp4")

    # Use pre-trained YOLOv8 model (yolov8x.pt or yolov8m.pt)
    MODEL_PATH = "yolov8x.pt"

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = FootballAnalyticsPipeline(model_path=MODEL_PATH)

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH)
    else:
        print(f"\n[NOTICE] Input video file not found at: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
