"""
Keypoint-Based Tactical Pitch Skeleton & Dynamic Polygon Pipeline (v3 - Corrected Topology)
Phase 1 - Football Performance Analysis System

Key Architectural Fixes:
1. Exact 32-Keypoint Topology Mapping for martinjolif/yolo-football-pitch-detection (DFL/SoccerNet Schema).
2. Corrected PITCH_SKELETON_EDGES to eliminate random diagonal lines across the pitch.
3. Hybrid Pitch Polygon Boundary (Combines Adaptive Green Turf Mask + Keypoint Hull) so the pitch
   overlay covers the full visible playing surface without truncating off-camera field areas.

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


# Exact 32-Keypoint Topology for martinjolif/yolo-football-pitch-detection Model
# Schema: [Left Side (0-12), Halfway Line (13-16, 31), Right Side (17-30)]
PITCH_SKELETON_EDGES_32 = [
    # --- Halfway Line & Center Circle ---
    (13, 14), (14, 15), (15, 31), (31, 16),
    (13, 16),

    # --- Right Goal Area & Penalty Box (Right Side) ---
    (17, 26), (26, 22), (22, 19), (19, 17),  # Right 18-yard box
    (18, 22), (19, 18),                      # Right 6-yard box
    (24, 25), (25, 26),                      # Right touchline / corner

    # --- Left Goal Area & Penalty Box (Left Side) ---
    (2, 9), (9, 10), (10, 5), (5, 2),        # Left 18-yard box
    (3, 7), (7, 8), (8, 4), (4, 3),          # Left 6-yard box
    (0, 1), (1, 2)                           # Left touchline / corner
]


class AdaptiveHSVGrassSegmentorV3:
    """
    Adaptive HSV green turf segmentor (v2 logic).
    """

    def __init__(self, lower_green=(32, 65, 65), upper_green=(85, 255, 255)):
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)

    def segment_turf(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Exclude upper 15% crowd area
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


class LabLineDetectorV3:
    """
    L*a*b* Color Space Line Detector (v2 logic).
    """

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


class KeypointPitchSegmentor:
    """
    Primary Engine: Keypoint-Based Pitch Landmark & Wireframe Detector.
    Uses custom YOLO Pose/Landmark weights when loaded.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.using_keypoints = False

        if model_path and os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                print(f"[INFO] Loading Pitch Keypoint model: {model_path}")
                self.model = YOLO(model_path)
                self.using_keypoints = True
            except Exception as e:
                print(f"[WARN] Failed to load Keypoint model: {e}")

    def detect_keypoints(self, frame: np.ndarray) -> Tuple[Dict[int, Tuple[int, int, float]], str]:
        if self.using_keypoints and self.model is not None:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
            if hasattr(results, 'keypoints') and results.keypoints is not None and len(results.keypoints) > 0:
                kpts_data = results.keypoints.data[0].cpu().numpy()  # Shape: (32, 3) -> x, y, conf
                keypoints_dict = {}
                for idx, (x, y, conf) in enumerate(kpts_data):
                    if conf >= self.conf_threshold:
                        keypoints_dict[idx] = (int(x), int(y), float(conf))

                if len(keypoints_dict) >= 4:
                    return keypoints_dict, "Keypoint Wireframe"

        return {}, "HSV Fallback"


class BoundaryExtractorV3:
    """
    Extracts outer pitch boundary polygon (Hybrid Turf + Keypoints).
    """

    @staticmethod
    def extract_hybrid_boundary(turf_mask: np.ndarray, keypoints_dict: Dict[int, Tuple[int, int, float]]) -> Tuple[Optional[np.ndarray], Optional[Polygon]]:
        """
        Combines Green Turf Mask contour with Keypoint Convex Hull to form a complete field polygon.
        """
        hybrid_mask = turf_mask.copy()

        if len(keypoints_dict) >= 3:
            pts = np.array([[x, y] for (x, y, c) in keypoints_dict.values()], dtype=np.int32)
            cv2.fillConvexPoly(hybrid_mask, pts, 255)

        contours, _ = cv2.findContours(hybrid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 2000:
            return None, None

        hull = cv2.convexHull(largest)
        epsilon = 0.012 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        poly_pts = approx.reshape(-1, 2)
        shapely_poly = None
        if len(poly_pts) >= 3:
            try:
                shapely_poly = Polygon(poly_pts)
                if not shapely_poly.is_valid:
                    shapely_poly = shapely_poly.buffer(0)
            except Exception:
                shapely_poly = None

        return approx, shapely_poly


class PitchVisualizerV3:
    """
    Renders Keypoint Markers, Connected Tactical Wireframe Edges,
    Pitch Boundary Polygon, Semi-Transparent Turf, and Status HUD.
    """

    def __init__(self, alpha: float = 0.30):
        self.alpha = alpha

    def draw(
        self,
        frame: np.ndarray,
        keypoints_dict: Dict[int, Tuple[int, int, float]],
        turf_mask: np.ndarray,
        line_mask: np.ndarray,
        boundary_contour: Optional[np.ndarray],
        engine_mode: str = "HSV Fallback",
        fps: float = 0.0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Full-Pitch Semi-Transparent Turf Overlay
        if np.count_nonzero(turf_mask) > 0:
            turf_color = np.zeros_like(frame)
            turf_color[turf_mask > 0] = (0, 200, 0)
            overlay = cv2.addWeighted(overlay, 1.0, turf_color, self.alpha, 0)

        # 2. Highlight Pitch Line Mask in Neon Yellow
        if line_mask is not None and np.count_nonzero(line_mask) > 0:
            overlay[line_mask > 0] = (0, 255, 255)

        # 3. Primary Engine: Draw Tactical Keypoint Wireframe (Corrected Edges)
        if keypoints_dict:
            # Draw Skeleton Connection Lines along true field markings
            for kp1_id, kp2_id in PITCH_SKELETON_EDGES_32:
                if kp1_id in keypoints_dict and kp2_id in keypoints_dict:
                    pt1 = (keypoints_dict[kp1_id][0], keypoints_dict[kp1_id][1])
                    pt2 = (keypoints_dict[kp2_id][0], keypoints_dict[kp2_id][1])
                    cv2.line(overlay, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)  # Bright green wireframe

            # Draw Keypoint Landmark Markers
            for kp_id, (x, y, conf) in keypoints_dict.items():
                cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)  # Red outer dot
                cv2.circle(overlay, (x, y), 3, (255, 255, 255), -1)  # White center dot
                cv2.putText(
                    overlay, str(kp_id), (x + 7, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1
                )

        # 4. Outer Boundary Polygon Overlay (Cyan)
        if boundary_contour is not None:
            cv2.drawContours(overlay, [boundary_contour], -1, (255, 255, 0), 3)

        # 5. Real-Time Engine Status HUD (Top Left Corner)
        active_kpts_count = len(keypoints_dict)
        if engine_mode == "Keypoint Wireframe":
            status_text = f"Engine: Keypoint Wireframe (Active: {active_kpts_count}) | FPS: {fps:.1f}"
            badge_color = (0, 150, 0)
        else:
            status_text = f"Engine: HSV Fallback (v2) | FPS: {fps:.1f}"
            badge_color = (0, 80, 200)

        cv2.rectangle(overlay, (10, 10), (480, 50), badge_color, -1)
        cv2.putText(overlay, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # 6. PIP Debug Mask View (Top Right Corner)
        pip_w, pip_h = w // 4, h // 4
        debug_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if np.count_nonzero(turf_mask) > 0:
            debug_rgb[turf_mask > 0] = (0, 120, 0)
        if line_mask is not None and np.count_nonzero(line_mask) > 0:
            debug_rgb[line_mask > 0] = (0, 255, 255)
        if boundary_contour is not None:
            cv2.drawContours(debug_rgb, [boundary_contour], -1, (255, 255, 0), 2)

        pip_resized = cv2.resize(debug_rgb, (pip_w, pip_h))
        cv2.rectangle(pip_resized, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)

        margin = 15
        overlay[margin:margin + pip_h, w - pip_w - margin:w - margin] = pip_resized
        cv2.putText(
            overlay, "PIP View (v3)", (w - pip_w - margin + 5, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

        return overlay


class PitchSegmentationPipelineV3:
    """
    Master Pipeline Class for Phase 1 (v3) Tactical Wireframe Architecture.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35):
        self.kp_segmentor = KeypointPitchSegmentor(model_path=model_path, conf_threshold=conf_threshold)
        self.turf_segmentor = AdaptiveHSVGrassSegmentorV3()
        self.line_detector = LabLineDetectorV3()
        self.boundary_extractor = BoundaryExtractorV3()
        self.visualizer = PitchVisualizerV3()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()

        # 1. Segment Turf & Lines Baseline
        turf_mask = self.turf_segmentor.segment_turf(frame)
        line_mask = self.line_detector.detect_lines(frame, turf_mask)

        # 2. Keypoint Detection
        keypoints_dict, engine_mode = self.kp_segmentor.detect_keypoints(frame)

        # 3. Hybrid Boundary Extraction (Combines Turf + Keypoints)
        boundary_contour, shapely_polygon = self.boundary_extractor.extract_hybrid_boundary(
            turf_mask=turf_mask, keypoints_dict=keypoints_dict
        )

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 4. Render Final Visualization
        rendered_frame = self.visualizer.draw(
            frame=frame,
            keypoints_dict=keypoints_dict,
            turf_mask=turf_mask,
            line_mask=line_mask,
            boundary_contour=boundary_contour,
            engine_mode=engine_mode,
            fps=fps,
        )

        meta = {
            "keypoints_dict": keypoints_dict,
            "boundary_contour": boundary_contour,
            "shapely_polygon": shapely_polygon,
            "engine_mode": engine_mode,
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
        print("    PITCH SKELETON WIREFRAME PIPELINE (v3)        ")
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
                    kpt_count = len(meta["keypoints_dict"])
                    print(
                        f"[PROGRESS v3] Frame {frame_count}/{total_frames} | "
                        f"Engine: {meta['engine_mode']} (Kpts: {kpt_count}) | FPS: {meta['fps']:.1f}"
                    )
        finally:
            cap.release()
            writer.release()

        print(f"\n[SUCCESS v3] Video processing complete. Output saved to: {output_path}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "pitch_keypoints_yolov8.pt")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pitch_segmentation_v3_output.mp4")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = PitchSegmentationPipelineV3(model_path=MODEL_PATH)

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH)
    else:
        print(f"\n[NOTICE] Input video clip not found at: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
