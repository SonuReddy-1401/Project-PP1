"""
Production-Grade Pitch Segmentation & Homography Engine (v6)
Phase 1 - Football Performance Analysis System

Key Architectural Fixes in v6:
1. Turf-Mask Constrained Line Projection (Mask Clipping): Projected homography lines are strictly
   clipped inside the green field mask (Projected_Lines AND Turf_Mask), preventing ANY lines from flying into crowds/stands.
2. Spatial Keypoint Spread & Outlier Guard: Validates keypoint spatial distribution. Rejects ill-conditioned
   homography matrices when keypoints are tightly clustered on one side.
3. Stable Camera Matrix Memory: Retains high-confidence perspective state during fast camera motion.
4. Dual-Engine Hybrid Visualizer: Displays clean projected lines, 2D top-down minimap, and HUD status.

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation
"""

import os
import time
from typing import Tuple, Optional, Dict, List, Any
import cv2
import numpy as np
from shapely.geometry import Polygon

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


PITCH_W = 105.0
PITCH_H = 68.0
CANVAS_W = 1050
CANVAS_H = 680

def m2pix(x_m: float, y_m: float) -> Tuple[int, int]:
    return int((x_m / PITCH_W) * CANVAS_W), int((y_m / PITCH_H) * CANVAS_H)

# Real-World 2D Keypoint Mapping (in meters)
PITCH_KEYPOINT_WORLD_METERS = {
    0: (0.0, 0.0), 1: (0.0, 13.84), 2: (16.5, 13.84), 3: (5.5, 24.84),
    4: (5.5, 43.16), 5: (16.5, 54.16), 6: (11.0, 34.0), 7: (0.0, 24.84),
    8: (0.0, 43.16), 9: (16.5, 13.84), 10: (16.5, 54.16), 11: (0.0, 68.0),
    12: (0.0, 54.16), 13: (52.5, 0.0), 14: (52.5, 24.85), 15: (52.5, 34.0),
    31: (52.5, 43.15), 16: (52.5, 68.0), 17: (88.5, 13.84), 18: (99.5, 24.84),
    19: (99.5, 43.16), 21: (94.0, 34.0), 22: (105.0, 24.84), 24: (105.0, 0.0),
    25: (105.0, 0.0), 26: (105.0, 13.84), 30: (0.0, 68.0)
}


class CanonicalPitchCanvasV6:
    """Draws top-down 2D tactical pitch canvas."""

    @staticmethod
    def draw_topdown_pitch() -> np.ndarray:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:, :] = (20, 120, 20)
        white, thick = (255, 255, 255), 2

        cv2.rectangle(canvas, (0, 0), (CANVAS_W - 1, CANVAS_H - 1), white, thick)
        mid_x = CANVAS_W // 2
        cv2.line(canvas, (mid_x, 0), (mid_x, CANVAS_H), white, thick)
        center_pt = (mid_x, CANVAS_H // 2)
        cv2.circle(canvas, center_pt, int(9.15 / PITCH_H * CANVAS_H), white, thick)
        cv2.circle(canvas, center_pt, 4, white, -1)

        w_18 = int((16.5 / PITCH_W) * CANVAS_W)
        y1_18 = int((13.84 / PITCH_H) * CANVAS_H)
        y2_18 = int((54.16 / PITCH_H) * CANVAS_H)
        cv2.rectangle(canvas, (0, y1_18), (w_18, y2_18), white, thick)
        cv2.rectangle(canvas, (CANVAS_W - w_18, y1_18), (CANVAS_W, y2_18), white, thick)

        w_6 = int((5.5 / PITCH_W) * CANVAS_W)
        y1_6 = int((24.84 / PITCH_H) * CANVAS_H)
        y2_6 = int((43.16 / PITCH_H) * CANVAS_H)
        cv2.rectangle(canvas, (0, y1_6), (w_6, y2_6), white, thick)
        cv2.rectangle(canvas, (CANVAS_W - w_6, y1_6), (CANVAS_W, y2_6), white, thick)

        cv2.circle(canvas, (int((11.0 / PITCH_W) * CANVAS_W), CANVAS_H // 2), 3, white, -1)
        cv2.circle(canvas, (int((94.0 / PITCH_W) * CANVAS_W), CANVAS_H // 2), 3, white, -1)

        return canvas


class AdaptiveHSVGrassSegmentorV6:
    """Refined Green Grass Turf Segmentor."""

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


class LabLineDetectorV6:
    """L*a*b* Color Space Line Detector."""

    def __init__(self, adaptive_block_size: int = 15, c_val: int = 5):
        self.adaptive_block_size = adaptive_block_size
        self.c_val = c_val
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def detect_lines(self, frame: np.ndarray, turf_mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(turf_mask) == 0:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self.clahe.apply(l_channel)

        adaptive_lines = cv2.adaptiveThreshold(
            enhanced_l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.adaptive_block_size, -self.c_val
        )

        color_dist = np.abs(a_channel.astype(np.float32) - 128) + np.abs(b_channel.astype(np.float32) - 128)
        neutral_mask = (color_dist < 25).astype(np.uint8) * 255

        lines = cv2.bitwise_and(adaptive_lines, neutral_mask)
        line_mask = cv2.bitwise_and(lines, lines, mask=turf_mask)

        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_line)


class RobustHomographyEngineV6:
    """
    Robust Homography Engine with Spatial Spread Guard, RANSAC Validation,
    and Exponential Moving Average (EMA) Temporal Memory.
    """

    def __init__(self, alpha: float = 0.20, frame_w: int = 1920, frame_h: int = 1080):
        self.alpha = alpha
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.H_stable: Optional[np.ndarray] = None

    def validate_spatial_spread(self, dst_pts: np.ndarray) -> bool:
        """
        Validates if keypoints have sufficient spatial spread across the screen.
        Rejects ill-conditioned point clusters concentrated on one corner.
        """
        if len(dst_pts) < 4:
            return False

        xs = dst_pts[:, 0]
        ys = dst_pts[:, 1]

        spread_x = (np.max(xs) - np.min(xs)) / self.frame_w
        spread_y = (np.max(ys) - np.min(ys)) / self.frame_h

        # Require at least 25% horizontal spread or 20% vertical spread
        return (spread_x >= 0.25) or (spread_y >= 0.20)

    def compute_homography(
        self, keypoints_dict: Dict[int, Tuple[int, int, float]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        src_pts, dst_pts = [], []

        for kp_id, (img_x, img_y, conf) in keypoints_dict.items():
            if kp_id in PITCH_KEYPOINT_WORLD_METERS and conf >= 0.35:
                world_x, world_y = PITCH_KEYPOINT_WORLD_METERS[kp_id]
                px, py = m2pix(world_x, world_y)
                src_pts.append([px, py])
                dst_pts.append([img_x, img_y])

        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)

            if self.validate_spatial_spread(dst_arr):
                H_raw, inliers = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)

                if H_raw is not None and not np.isnan(H_raw).any():
                    num_inliers = int(np.sum(inliers)) if inliers is not None else 0

                    if num_inliers >= 4:
                        if self.H_stable is None:
                            self.H_stable = H_raw
                        else:
                            # EMA Temporal Smoothing
                            self.H_stable = (self.alpha * H_raw) + ((1.0 - self.alpha) * self.H_stable)

                        try:
                            H_inv = np.linalg.inv(self.H_stable)
                            return self.H_stable, H_inv, "RANSAC Valid"
                        except np.linalg.LinAlgError:
                            pass

        # Fallback to stable memory during ill-conditioned frames or dropouts
        if self.H_stable is not None:
            try:
                H_inv = np.linalg.inv(self.H_stable)
                return self.H_stable, H_inv, "Stable Memory"
            except np.linalg.LinAlgError:
                return None, None, "Failed"

        return None, None, "Searching"


class PitchVisualizerV6:
    """
    Visualizer featuring Turf-Mask Constrained Projection (Mask Clipping).
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.topdown_canvas = CanonicalPitchCanvasV6.draw_topdown_pitch()

    def draw(
        self,
        frame: np.ndarray,
        keypoints_dict: Dict[int, Tuple[int, int, float]],
        turf_mask: np.ndarray,
        line_mask: np.ndarray,
        H: Optional[np.ndarray],
        H_inv: Optional[np.ndarray],
        h_status: str,
        fps: float = 0.0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Semi-Transparent Green Turf Overlay
        if np.count_nonzero(turf_mask) > 0:
            turf_color = np.zeros_like(frame)
            turf_color[turf_mask > 0] = (0, 200, 0)
            overlay = cv2.addWeighted(overlay, 1.0, turf_color, self.alpha, 0)

        # 2. Highlight Physical L*a*b* Pitch Lines in Bright Yellow
        if line_mask is not None and np.count_nonzero(line_mask) > 0:
            overlay[line_mask > 0] = (0, 255, 255)

        # 3. Homography Projected Pitch Lines (STRICTLY CLIPPED INSIDE TURF MASK)
        if H is not None and np.count_nonzero(turf_mask) > 0:
            warped_canvas = cv2.warpPerspective(self.topdown_canvas, H, (w, h))

            # Raw Homography line pixels
            raw_lines = (warped_canvas[:, :, 0] > 200) & (warped_canvas[:, :, 1] > 200) & (warped_canvas[:, :, 2] > 200)

            # MASK CLIPPING: Force lines to ONLY exist inside the green turf mask!
            clipped_lines = raw_lines & (turf_mask > 0)
            overlay[clipped_lines] = (0, 255, 255)  # Neon Yellow projected lines

        # 4. Keypoint Markers
        for kp_id, (x, y, conf) in keypoints_dict.items():
            if conf >= 0.35:
                cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 255), -1)

        # 5. Real-Time Status HUD Overlay
        active_kpts = len(keypoints_dict)
        if h_status in ["RANSAC Valid", "Stable Memory"]:
            status_text = f"Engine: Homography (Mask Clipped | {h_status}) | Kpts: {active_kpts} | FPS: {fps:.1f}"
            badge_color = (0, 140, 0)
        else:
            status_text = f"Engine: Adaptive HSV+Lab (v2 Fallback) | FPS: {fps:.1f}"
            badge_color = (0, 80, 200)

        cv2.rectangle(overlay, (10, 10), (620, 50), badge_color, -1)
        cv2.putText(overlay, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 6. 2D Top-Down Minimap (PIP - Top Right Corner)
        pip_w, pip_h = w // 4, h // 4
        minimap = self.topdown_canvas.copy()

        if H_inv is not None:
            corners_img = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
            corners_canvas = cv2.perspectiveTransform(corners_img, H_inv)
            pts_canvas = corners_canvas.reshape(-1, 2).astype(np.int32)
            cv2.polylines(minimap, [pts_canvas], isClosed=True, color=(0, 255, 255), thickness=3)

        pip_resized = cv2.resize(minimap, (pip_w, pip_h))
        cv2.rectangle(pip_resized, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)

        margin = 15
        overlay[margin:margin + pip_h, w - pip_w - margin:w - margin] = pip_resized
        cv2.putText(
            overlay, "2D Tactical Radar (v6)", (w - pip_w - margin + 5, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

        return overlay


class PitchSegmentationPipelineV6:
    """
    Master Pipeline Class for Phase 1 (v6) Production Engine.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.turf_segmentor = AdaptiveHSVGrassSegmentorV6()
        self.line_detector = LabLineDetectorV6()
        self.homography_engine = RobustHomographyEngineV6()
        self.visualizer = PitchVisualizerV6()
        self.model = None
        self.using_keypoints = False

        if model_path and os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                print(f"[INFO] Loading Pitch Keypoint Model: {model_path}")
                self.model = YOLO(model_path)
                self.using_keypoints = True
            except Exception as e:
                print(f"[WARN] Failed to load Keypoint model: {e}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()

        # 1. Turf & Line Mask Baselines
        turf_mask = self.turf_segmentor.segment_turf(frame)
        line_mask = self.line_detector.detect_lines(frame, turf_mask)

        # 2. YOLO Keypoint Detections
        keypoints_dict = {}
        if self.using_keypoints and self.model is not None:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
            if hasattr(results, 'keypoints') and results.keypoints is not None and len(results.keypoints) > 0:
                kpts_data = results.keypoints.data[0].cpu().numpy()
                for idx, (x, y, conf) in enumerate(kpts_data):
                    if conf >= self.conf_threshold:
                        keypoints_dict[idx] = (int(x), int(y), float(conf))

        # 3. Robust Homography Matrix Calculation with Spatial Spread Guard
        H, H_inv, h_status = self.homography_engine.compute_homography(keypoints_dict)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 4. Render Final Mask-Clipped Output Frame
        rendered_frame = self.visualizer.draw(
            frame=frame,
            keypoints_dict=keypoints_dict,
            turf_mask=turf_mask,
            line_mask=line_mask,
            H=H,
            H_inv=H_inv,
            h_status=h_status,
            fps=fps,
        )

        meta = {
            "keypoints_dict": keypoints_dict,
            "H_matrix": H,
            "H_inv_matrix": H_inv,
            "h_status": h_status,
            "fps": fps,
        }

        return rendered_frame, meta

    def process_video(self, video_path: str, output_path: str) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("==================================================")
        print("  PRODUCTION PITCH HOMOGRAPHY ENGINE (v6)        ")
        print("==================================================")
        print(f"[INFO] Video File: {video_path}")
        print(f"[INFO] Resolution: {width}x{height} | FPS: {input_fps:.2f} | Total Frames: {total_frames}\n")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))

        frame_count = 0
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
                        f"[PROGRESS v6] Frame {frame_count}/{total_frames} | "
                        f"Status: {meta['h_status']} (Kpts: {len(meta['keypoints_dict'])}) | FPS: {meta['fps']:.1f}"
                    )
        finally:
            cap.release()
            writer.release()

        print(f"\n[SUCCESS v6] Processing complete. Output saved to: {output_path}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "pitch_keypoints_yolov8.pt")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pitch_segmentation_v6_output.mp4")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = PitchSegmentationPipelineV6(model_path=MODEL_PATH)

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH)
    else:
        print(f"\n[NOTICE] Input video clip not found at: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
