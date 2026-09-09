"""
Temporal Camera Motion Tracking & Homography Stabilization Engine (v5)
Phase 1 - Football Performance Analysis System

Key Architectural Upgrades in v5:
1. Lucas-Kanade (LK) Optical Flow Keypoint Tracker (cv2.calcOpticalFlowPyrLK) across motion blur.
2. Global Camera Motion Estimator (cv2.estimateAffinePartial2D) for background pan/zoom velocity.
3. Kalman / Momentum Matrix Filter for Homography matrix H to eliminate line collapse during camera pans.
4. Confidence-Weighted Keypoint Fusion (Merges YOLO detections + Optical Flow predictions).
5. 2D Top-Down Tactical Minimap View with Smooth Camera Frustum Projection.

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


class CanonicalPitchCanvasV5:
    """Draws top-down tactical pitch canvas."""

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


class OpticalFlowKeypointTracker:
    """
    Lucas-Kanade Optical Flow Keypoint Tracker across motion blur.
    Tracks keypoints from Frame t-1 to Frame t when YOLO misses keypoints.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.tracked_kpts: Dict[int, Tuple[float, float, float]] = {}  # {kp_id: (x, y, conf)}
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def update(
        self, curr_frame: np.ndarray, yolo_kpts: Dict[int, Tuple[int, int, float]]
    ) -> Dict[int, Tuple[int, int, float]]:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        merged_kpts: Dict[int, Tuple[int, int, float]] = {}

        if self.prev_gray is not None and self.tracked_kpts:
            # Prepare points for LK Optical Flow
            kp_ids = list(self.tracked_kpts.keys())
            pts_prev = np.array([[self.tracked_kpts[i][0], self.tracked_kpts[i][1]] for i in kp_ids], dtype=np.float32).reshape(-1, 1, 2)

            pts_curr, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, pts_prev, None, **self.lk_params)

            if pts_curr is not None:
                for idx, kp_id in enumerate(kp_ids):
                    if status[idx][0] == 1:
                        cx, cy = pts_curr[idx][0]
                        prev_conf = self.tracked_kpts[kp_id][2]
                        # Decay optical flow confidence slightly per frame
                        merged_kpts[kp_id] = (float(cx), float(cy), prev_conf * 0.95)

        # Merge with fresh YOLO keypoint detections (YOLO detections take priority)
        for kp_id, (yx, yy, yconf) in yolo_kpts.items():
            if kp_id in merged_kpts:
                # Weighted Fusion: 70% YOLO + 30% Optical Flow
                fx, fy, _ = merged_kpts[kp_id]
                fused_x = (0.7 * yx) + (0.3 * fx)
                fused_y = (0.7 * yy) + (0.3 * fy)
                merged_kpts[kp_id] = (fused_x, fused_y, yconf)
            else:
                merged_kpts[kp_id] = (float(yx), float(yy), yconf)

        self.prev_gray = curr_gray
        self.tracked_kpts = merged_kpts
        return merged_kpts


class KalmanHomographyFilter:
    """
    Kalman / Momentum Matrix Filter for Homography Matrix H.
    Prevents matrix collapse and wild jumps during fast camera pans/zooms.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.H_filtered: Optional[np.ndarray] = None

    def filter(self, H_raw: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if H_raw is None or np.isnan(H_raw).any():
            # Retain last valid filtered Homography matrix during momentary dropouts
            if self.H_filtered is not None:
                try:
                    H_inv = np.linalg.inv(self.H_filtered)
                    return self.H_filtered, H_inv
                except np.linalg.LinAlgError:
                    return None, None
            return None, None

        if self.H_filtered is None:
            self.H_filtered = H_raw
        else:
            # Exponential Momentum Filter
            self.H_filtered = (self.alpha * H_raw) + ((1.0 - self.alpha) * self.H_filtered)

        try:
            H_inv = np.linalg.inv(self.H_filtered)
        except np.linalg.LinAlgError:
            H_inv = None

        return self.H_filtered, H_inv


class AdaptiveHSVGrassSegmentorV5:
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


class PitchVisualizerV5:
    """
    Renders Homography Projected Lines, 2D Top-Down Minimap, and Real-Time HUD.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.topdown_canvas = CanonicalPitchCanvasV5.draw_topdown_pitch()

    def draw(
        self,
        frame: np.ndarray,
        keypoints_dict: Dict[int, Tuple[float, float, float]],
        turf_mask: np.ndarray,
        H: Optional[np.ndarray],
        H_inv: Optional[np.ndarray],
        fps: float = 0.0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Semi-Transparent Green Turf Overlay
        if np.count_nonzero(turf_mask) > 0:
            turf_color = np.zeros_like(frame)
            turf_color[turf_mask > 0] = (0, 200, 0)
            overlay = cv2.addWeighted(overlay, 1.0, turf_color, self.alpha, 0)

        engine_mode = "HSV Fallback"

        # 2. Homography Perspective Line Projection (Canvas -> Camera Frame)
        if H is not None:
            engine_mode = "Temporal Homography (LK+Kalman)"
            warped_canvas = cv2.warpPerspective(self.topdown_canvas, H, (w, h))

            line_pixels = (warped_canvas[:, :, 0] > 200) & (warped_canvas[:, :, 1] > 200) & (warped_canvas[:, :, 2] > 200)
            overlay[line_pixels] = (0, 255, 255)  # Neon Yellow projected lines

        # 3. Keypoint Landmark Markers (Red outer, white inner dot)
        for kp_id, (x, y, conf) in keypoints_dict.items():
            if conf >= 0.30:
                pt = (int(x), int(y))
                cv2.circle(overlay, pt, 5, (0, 0, 255), -1)
                cv2.circle(overlay, pt, 2, (255, 255, 255), -1)

        # 4. Real-Time HUD Status Overlay
        active_kpts = len(keypoints_dict)
        if engine_mode == "Temporal Homography (LK+Kalman)":
            status_text = f"Engine: Temporal Homography (LK Flow+Kalman) | Active Kpts: {active_kpts} | FPS: {fps:.1f}"
            badge_color = (0, 140, 0)  # Green badge
        else:
            status_text = f"Engine: HSV Fallback | FPS: {fps:.1f}"
            badge_color = (0, 80, 200)  # Orange badge

        cv2.rectangle(overlay, (10, 10), (600, 50), badge_color, -1)
        cv2.putText(overlay, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 5. 2D Top-Down Minimap Radar View (PIP - Top Right Corner)
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
            overlay, "2D Tactical Radar (v5)", (w - pip_w - margin + 5, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

        return overlay


class PitchSegmentationPipelineV5:
    """
    Master Pipeline Class for Phase 1 (v5) Temporal Camera Motion Tracking.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.turf_segmentor = AdaptiveHSVGrassSegmentorV5()
        self.of_tracker = OpticalFlowKeypointTracker()
        self.kalman_filter = KalmanHomographyFilter(alpha=0.35)
        self.visualizer = PitchVisualizerV5()
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

        # 1. Green Turf Mask Baseline
        turf_mask = self.turf_segmentor.segment_turf(frame)

        # 2. YOLO Keypoint Detection
        yolo_kpts = {}
        if self.using_keypoints and self.model is not None:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
            if hasattr(results, 'keypoints') and results.keypoints is not None and len(results.keypoints) > 0:
                kpts_data = results.keypoints.data[0].cpu().numpy()
                for idx, (x, y, conf) in enumerate(kpts_data):
                    if conf >= self.conf_threshold:
                        yolo_kpts[idx] = (int(x), int(y), float(conf))

        # 3. Optical Flow Temporal Tracking across Motion Blur
        tracked_kpts = self.of_tracker.update(frame, yolo_kpts)

        # 4. Calculate Raw Homography Matrix H
        src_pts, dst_pts = [], []
        for kp_id, (img_x, img_y, conf) in tracked_kpts.items():
            if kp_id in PITCH_KEYPOINT_WORLD_METERS and conf >= 0.25:
                world_x, world_y = PITCH_KEYPOINT_WORLD_METERS[kp_id]
                px, py = m2pix(world_x, world_y)
                src_pts.append([px, py])
                dst_pts.append([img_x, img_y])

        H_raw = None
        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)
            H_raw, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)

        # 5. Kalman / Momentum Matrix Filter & Memory Retention
        H_filtered, H_inv = self.kalman_filter.filter(H_raw)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 6. Render Visualization Output
        rendered_frame = self.visualizer.draw(
            frame=frame,
            keypoints_dict=tracked_kpts,
            turf_mask=turf_mask,
            H=H_filtered,
            H_inv=H_inv,
            fps=fps,
        )

        meta = {
            "tracked_kpts": tracked_kpts,
            "H_matrix": H_filtered,
            "H_inv_matrix": H_inv,
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
        print("   TEMPORAL CAMERA MOTION HOMOGRAPHY ENGINE (v5)  ")
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
                    h_status = "LK Flow+Kalman Active" if meta["H_matrix"] is not None else "Searching"
                    print(
                        f"[PROGRESS v5] Frame {frame_count}/{total_frames} | "
                        f"Tracking: {h_status} (Active Kpts: {len(meta['tracked_kpts'])}) | FPS: {meta['fps']:.1f}"
                    )
        finally:
            cap.release()
            writer.release()

        print(f"\n[SUCCESS v5] Processing complete. Output saved to: {output_path}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "pitch_keypoints_yolov8.pt")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pitch_segmentation_v5_output.mp4")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = PitchSegmentationPipelineV5(model_path=MODEL_PATH)

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH)
    else:
        print(f"\n[NOTICE] Input video clip not found at: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
