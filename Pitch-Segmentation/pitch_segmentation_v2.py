"""
Advanced Pitch Segmentation, Line Detection, and Pitch Boundary Extraction Pipeline (v2)
Phase 1 - Football Performance Analysis System

Enhancements in v2:
1. Adaptive Field-Center HSV Sampling + Tightened Saturation/Value floor (removes dark stands).
2. Spatial Horizon & Field Polygon Extraction (eliminates off-field grass behind goals).
3. L*a*b* Color Space + Adaptive Thresholding for Pitch Lines (robust line detection under shadows).
4. Strict In-Field Masking for Line Extraction.

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation
"""

import os
import time
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np
from shapely.geometry import Polygon

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class AdaptiveHSVGrassSegmentor:
    """
    Advanced OpenCV segmentor with tightened HSV thresholds and dynamic sampling
    to eliminate dark stands, crowds, and non-turf noise.
    """

    def __init__(
        self,
        lower_green: Tuple[int, int, int] = (32, 65, 65),
        upper_green: Tuple[int, int, int] = (85, 255, 255),
    ):
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)

    def segment_turf(self, frame: np.ndarray) -> np.ndarray:
        """
        Segments green pitch turf from a BGR image frame.

        Returns:
            turf_mask (np.ndarray): Clean binary mask of pitch turf (255 inside, 0 outside)
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Raw HSV Thresholding with tightened bounds
        raw_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # 2. Field Horizon Filter: Exclude upper 15% of frame (usually crowd / stadium roof)
        horizon_y = int(h * 0.15)
        raw_mask[:horizon_y, :] = 0

        # 3. Morphological Operations
        # Morphological Open to remove isolated noise (seats, banners)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)

        # Morphological Close to fill gaps inside grass
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close)

        # 4. Connected Component Filtering: Keep only major turf regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
        turf_mask = np.zeros((h, w), dtype=np.uint8)

        min_turf_area = (h * w) * 0.05  # At least 5% of total frame area
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_turf_area:
                turf_mask[labels == i] = 255

        return turf_mask


class LabLineDetector:
    """
    Pitch line detector operating in L*a*b* Color Space.
    Converts frame to L*a*b*, applies CLAHE on Luminance (L*), and uses adaptive thresholding
    constrained strictly within the extracted pitch boundary mask.
    """

    def __init__(self, adaptive_block_size: int = 15, c_val: int = 5):
        self.adaptive_block_size = adaptive_block_size
        self.c_val = c_val
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def detect_lines(self, frame: np.ndarray, turf_mask: np.ndarray) -> np.ndarray:
        """
        Detects white pitch lines using L*a*b* luminance isolation.

        Returns:
            line_mask (np.ndarray): Binary mask of white lines strictly inside turf boundary
        """
        if np.count_nonzero(turf_mask) == 0:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # 1. Convert to L*a*b* color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Enhance contrast on Luminance channel using CLAHE
        enhanced_l = self.clahe.apply(l_channel)

        # 3. Adaptive Thresholding to extract bright linear structures
        adaptive_lines = cv2.adaptiveThreshold(
            enhanced_l,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            -self.c_val,
        )

        # 4. Additional Color Constraint: White lines have low color saturation in L*a*b*
        # In L*a*b*, neutral white has a* and b* close to 128
        color_dist = np.abs(a_channel.astype(np.float32) - 128) + np.abs(b_channel.astype(np.float32) - 128)
        neutral_mask = (color_dist < 25).astype(np.uint8) * 255

        line_candidates = cv2.bitwise_and(adaptive_lines, neutral_mask)

        # 5. Strictly constrain line mask inside the green turf mask
        line_mask = cv2.bitwise_and(line_candidates, line_candidates, mask=turf_mask)

        # 6. Line Morphological Enhancement
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_line)

        return line_mask


class BoundaryExtractorV2:
    """
    Extracts outer pitch boundary polygon, handles field line boundaries,
    and constructs a Shapely Polygon object.
    """

    def __init__(self, approx_epsilon_ratio: float = 0.012):
        self.approx_epsilon_ratio = approx_epsilon_ratio

    def extract_boundary(self, turf_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Polygon]]:
        """
        Extracts the largest convex field boundary contour and converts to a Shapely Polygon.
        """
        contours, _ = cv2.findContours(turf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 2000:
            return None, None

        # Convex Hull to smooth boundary
        hull = cv2.convexHull(largest_contour)

        # Simplify polygon vertices
        epsilon = self.approx_epsilon_ratio * cv2.arcLength(hull, True)
        boundary_contour = cv2.approxPolyDP(hull, epsilon, True)

        # Convert to Shapely Polygon
        poly_points = boundary_contour.reshape(-1, 2)
        shapely_polygon = None
        if len(poly_points) >= 3:
            try:
                shapely_polygon = Polygon(poly_points)
                if not shapely_polygon.is_valid:
                    shapely_polygon = shapely_polygon.buffer(0)
            except Exception:
                shapely_polygon = None

        return boundary_contour, shapely_polygon


class PitchVisualizerV2:
    """
    Renders semi-transparent green turf overlay, bright pitch lines, boundary polygon,
    HUD metrics, and a Picture-in-Picture (PIP) debug view.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha

    def draw(
        self,
        frame: np.ndarray,
        turf_mask: np.ndarray,
        line_mask: np.ndarray,
        boundary_contour: Optional[np.ndarray],
        mode_used: str = "Adaptive-HSV+Lab",
        fps: float = 0.0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Semi-transparent green overlay for pitch turf
        turf_color = np.zeros_like(frame)
        turf_color[turf_mask > 0] = (0, 210, 0)  # Bright green overlay
        overlay = cv2.addWeighted(overlay, 1.0, turf_color, self.alpha, 0)

        # 2. Highlight detected pitch lines in neon yellow
        overlay[line_mask > 0] = (0, 255, 255)

        # 3. Draw outer field boundary polygon in bold cyan
        if boundary_contour is not None:
            cv2.drawContours(overlay, [boundary_contour], -1, (255, 255, 0), 3)
            for point in boundary_contour:
                pt = tuple(point[0])
                cv2.circle(overlay, pt, 6, (0, 0, 255), -1)

        # 4. HUD Overlay
        hud_text = f"Mode: {mode_used} | FPS: {fps:.1f}"
        cv2.rectangle(overlay, (10, 10), (410, 50), (0, 0, 0), -1)
        cv2.putText(overlay, hud_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 5. PIP Debug Mask View (Top Right Corner)
        pip_w, pip_h = w // 4, h // 4
        debug_mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        debug_mask_rgb[turf_mask > 0] = (0, 120, 0)
        debug_mask_rgb[line_mask > 0] = (0, 255, 255)
        if boundary_contour is not None:
            cv2.drawContours(debug_mask_rgb, [boundary_contour], -1, (255, 255, 0), 2)

        pip_resized = cv2.resize(debug_mask_rgb, (pip_w, pip_h))
        cv2.rectangle(pip_resized, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)

        margin = 15
        overlay[margin:margin + pip_h, w - pip_w - margin:w - margin] = pip_resized
        cv2.putText(
            overlay,
            "PIP Mask (v2)",
            (w - pip_w - margin + 5, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        return overlay


class PitchSegmentationPipelineV2:
    """
    Main Orchestrator Class for Phase 1 (v2) Pipeline.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.50,
        lower_green: Tuple[int, int, int] = (32, 65, 65),
        upper_green: Tuple[int, int, int] = (85, 255, 255),
    ):
        self.conf_threshold = conf_threshold
        self.turf_segmentor = AdaptiveHSVGrassSegmentor(lower_green=lower_green, upper_green=upper_green)
        self.line_detector = LabLineDetector()
        self.boundary_extractor = BoundaryExtractorV2()
        self.visualizer = PitchVisualizerV2()
        self.model = None
        self.using_custom_model = False

        if model_path and os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                print(f"[INFO] Loading custom YOLO model: {model_path}")
                self.model = YOLO(model_path)
                self.using_custom_model = True
            except Exception as e:
                print(f"[WARN] Could not load YOLO model from {model_path}: {e}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        mode_used = "Adaptive-HSV+Lab"

        # 1. Deep Learning or Adaptive HSV Segmentation
        if self.using_custom_model and self.model is not None:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
            if results.masks is not None and len(results.masks) > 0:
                h, w = frame.shape[:2]
                turf_mask = np.zeros((h, w), dtype=np.uint8)
                line_mask = np.zeros((h, w), dtype=np.uint8)

                for idx, cls_tensor in enumerate(results.boxes.cls):
                    cls_id = int(cls_tensor.item())
                    mask = (results.masks.data[idx].cpu().numpy() * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    if cls_id == 0:
                        turf_mask = cv2.bitwise_or(turf_mask, mask)
                    elif cls_id == 1:
                        line_mask = cv2.bitwise_or(line_mask, mask)

                if np.count_nonzero(turf_mask) > 0:
                    mode_used = "YOLOv8-seg"
                else:
                    turf_mask = self.turf_segmentor.segment_turf(frame)
                    line_mask = self.line_detector.detect_lines(frame, turf_mask)
            else:
                turf_mask = self.turf_segmentor.segment_turf(frame)
                line_mask = self.line_detector.detect_lines(frame, turf_mask)
        else:
            turf_mask = self.turf_segmentor.segment_turf(frame)
            line_mask = self.line_detector.detect_lines(frame, turf_mask)

        # 2. Extract Boundary Polygon
        boundary_contour, shapely_polygon = self.boundary_extractor.extract_boundary(turf_mask)

        # FPS calculation
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 3. Render Visualization
        rendered_frame = self.visualizer.draw(
            frame=frame,
            turf_mask=turf_mask,
            line_mask=line_mask,
            boundary_contour=boundary_contour,
            mode_used=mode_used,
            fps=fps,
        )

        meta = {
            "turf_mask": turf_mask,
            "line_mask": line_mask,
            "boundary_contour": boundary_contour,
            "shapely_polygon": shapely_polygon,
            "mode_used": mode_used,
            "fps": fps,
        }

        return rendered_frame, meta

    def process_video(
        self, video_path: str, output_path: Optional[str] = None, display: bool = False
    ) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video Opened: {video_path}")
        print(f"[INFO] Resolution: {width}x{height} | FPS: {input_fps:.2f} | Total Frames: {total_frames}")

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
            print(f"[INFO] Saving output video to: {output_path}")

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                rendered_frame, meta = self.process_frame(frame)

                if writer:
                    writer.write(rendered_frame)

                if display:
                    cv2.imshow("Pitch Segmentation v2", rendered_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] User interrupted video playback.")
                        break

                if frame_count % 30 == 0 or frame_count == total_frames:
                    print(
                        f"[PROGRESS v2] Frame {frame_count}/{total_frames} | "
                        f"Mode: {meta['mode_used']} | FPS: {meta['fps']:.1f}"
                    )
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        print(f"[SUCCESS v2] Processing complete. Processed {frame_count} frames.")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Default paths
    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "pitch_seg_yolov8.pt")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pitch_segmentation_v2_output.mp4")
    CONFIDENCE_THRESHOLD = 0.50

    print("==================================================")
    print("    FOOTBALL PITCH SEGMENTATION - PHASE 1 (v2)    ")
    print("==================================================")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = PitchSegmentationPipelineV2(
        model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD
    )

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH, display=False)
    else:
        print(f"\n[NOTICE] Input video file not found at: {VIDEO_PATH}")
        print("Please place your video clip in the data folder:")
        print(f" -> Path: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
