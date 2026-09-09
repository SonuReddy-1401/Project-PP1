"""
Pitch Segmentation, Line Detection, and Pitch Boundary Extraction Pipeline
Phase 1 - Football Performance Analysis System

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


class HSVGrassSegmentor:
    """
    Pure OpenCV color-space fallback segmentor.
    Extracts green playing turf and white pitch lines dynamically using HSV thresholding.
    """

    def __init__(
        self,
        lower_green: Tuple[int, int, int] = (30, 40, 40),
        upper_green: Tuple[int, int, int] = (85, 255, 255),
        line_sat_max: int = 60,
        line_val_min: int = 170,
    ):
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)
        self.line_sat_max = line_sat_max
        self.line_val_min = line_val_min

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segments green pitch turf and white lines from a BGR image frame.

        Returns:
            turf_mask (np.ndarray): Binary mask (255 where turf exists, 0 elsewhere)
            line_mask (np.ndarray): Binary mask (255 where pitch lines exist, 0 elsewhere)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Green Turf Mask
        raw_turf_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Clean noise on turf mask
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

        turf_mask = cv2.morphologyEx(raw_turf_mask, cv2.MORPH_OPEN, kernel_open)
        turf_mask = cv2.morphologyEx(turf_mask, cv2.MORPH_CLOSE, kernel_close)

        # 2. White Line Mask (Constrained inside green turf region)
        # White has low saturation and high value
        line_raw_mask = (hsv[:, :, 1] <= self.line_sat_max) & (hsv[:, :, 2] >= self.line_val_min)
        line_raw_mask = (line_raw_mask.astype(np.uint8)) * 255

        # Constrain lines strictly inside the detected turf mask
        line_mask = cv2.bitwise_and(line_raw_mask, line_raw_mask, mask=turf_mask)

        # Morphological line enhancement
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_line)

        return turf_mask, line_mask


class PitchSegmentor:
    """
    Dual-Engine Segmentation Model Handling.
    Uses custom fine-tuned YOLOv8-seg when provided, falling back to HSVGrassSegmentor.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.50,
        hsv_segmentor: Optional[HSVGrassSegmentor] = None,
    ):
        self.conf_threshold = conf_threshold
        self.hsv_segmentor = hsv_segmentor or HSVGrassSegmentor()
        self.model = None
        self.using_custom_model = False

        if model_path and os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                print(f"[INFO] Loading custom YOLO segmentation model: {model_path}")
                self.model = YOLO(model_path)
                self.using_custom_model = True
            except Exception as e:
                print(f"[WARN] Failed to load model from {model_path}: {e}. Falling back to HSV.")

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Segments frame into turf and line masks.

        Returns:
            turf_mask (np.ndarray): Binary mask for pitch turf
            line_mask (np.ndarray): Binary mask for white pitch lines
            mode_used (str): 'YOLOv8-seg' or 'HSV-Fallback'
        """
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

                    # Class 0: pitch_turf, Class 1: pitch_lines (standard convention)
                    if cls_id == 0:
                        turf_mask = cv2.bitwise_or(turf_mask, mask)
                    elif cls_id == 1:
                        line_mask = cv2.bitwise_or(line_mask, mask)

                if np.count_nonzero(turf_mask) > 0:
                    return turf_mask, line_mask, "YOLOv8-seg"

        # Fallback to HSV color-space segmentor
        turf_mask, line_mask = self.hsv_segmentor.segment(frame)
        return turf_mask, line_mask, "HSV-Fallback"


class BoundaryExtractor:
    """
    Extracts outer pitch boundary contour and converts to a Shapely Polygon.
    """

    def __init__(self, approx_epsilon_ratio: float = 0.015):
        self.approx_epsilon_ratio = approx_epsilon_ratio

    def extract_boundary(self, turf_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Polygon]]:
        """
        Finds the largest external contour of the turf mask, applies convex hull / polygon approximation,
        and constructs a Shapely Polygon object.

        Returns:
            boundary_contour (np.ndarray or None): (N, 1, 2) array of boundary points
            shapely_polygon (Polygon or None): Polygon object for spatial operations
        """
        contours, _ = cv2.findContours(turf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Select largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 1000:  # Ignore tiny noise contours
            return None, None

        # Convex Hull to smooth boundary
        hull = cv2.convexHull(largest_contour)

        # Simplify polygon
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


class PitchVisualizer:
    """
    Visualizes segmented turf, pitch lines, boundary polygon, and PIP debug frame.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha

    def draw(
        self,
        frame: np.ndarray,
        turf_mask: np.ndarray,
        line_mask: np.ndarray,
        boundary_contour: Optional[np.ndarray],
        mode_used: str = "HSV-Fallback",
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Renders semi-transparent turf overlay, bright pitch lines, boundary polygon,
        HUD metrics, and a Picture-in-Picture (PIP) debug view.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Semi-transparent green overlay for pitch turf
        turf_color = np.zeros_like(frame)
        turf_color[turf_mask > 0] = (0, 200, 0)  # Bright green
        overlay = cv2.addWeighted(overlay, 1.0, turf_color, self.alpha, 0)

        # 2. Highlight detected pitch lines in neon yellow
        overlay[line_mask > 0] = (0, 255, 255)

        # 3. Draw outer field boundary polygon in bold cyan
        if boundary_contour is not None:
            cv2.drawContours(overlay, [boundary_contour], -1, (255, 255, 0), 3)
            # Draw vertex keypoints
            for point in boundary_contour:
                pt = tuple(point[0])
                cv2.circle(overlay, pt, 5, (0, 0, 255), -1)

        # 4. HUD Information Overlay
        hud_text = f"Mode: {mode_used} | FPS: {fps:.1f}"
        cv2.rectangle(overlay, (10, 10), (380, 50), (0, 0, 0), -1)
        cv2.putText(overlay, hud_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 5. Picture-in-Picture (PIP) Debug Mask (Top Right Corner)
        pip_w, pip_h = w // 4, h // 4
        debug_mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        debug_mask_rgb[turf_mask > 0] = (0, 120, 0)
        debug_mask_rgb[line_mask > 0] = (0, 255, 255)
        if boundary_contour is not None:
            cv2.drawContours(debug_mask_rgb, [boundary_contour], -1, (255, 255, 0), 2)

        pip_resized = cv2.resize(debug_mask_rgb, (pip_w, pip_h))
        # Add PIP border
        cv2.rectangle(pip_resized, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)

        # Overlay PIP onto top-right corner
        margin = 15
        overlay[margin:margin + pip_h, w - pip_w - margin:w - margin] = pip_resized
        cv2.putText(
            overlay,
            "PIP Mask View",
            (w - pip_w - margin + 5, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        return overlay


class PitchSegmentationPipeline:
    """
    Main Orchestrator Class for Phase 1 Pitch Segmentation Pipeline.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.50,
        lower_green: Tuple[int, int, int] = (30, 40, 40),
        upper_green: Tuple[int, int, int] = (85, 255, 255),
    ):
        self.hsv_segmentor = HSVGrassSegmentor(lower_green=lower_green, upper_green=upper_green)
        self.pitch_segmentor = PitchSegmentor(
            model_path=model_path, conf_threshold=conf_threshold, hsv_segmentor=self.hsv_segmentor
        )
        self.boundary_extractor = BoundaryExtractor()
        self.visualizer = PitchVisualizer()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR frame.

        Returns:
            rendered_frame (np.ndarray): Output visualization frame with overlays & HUD
            meta (dict): Dictionary containing raw masks, polygon, and mode details
        """
        start_time = time.time()

        # 1. Segment Turf and Lines
        turf_mask, line_mask, mode_used = self.pitch_segmentor.segment(frame)

        # 2. Extract Boundary Polygon
        boundary_contour, shapely_polygon = self.boundary_extractor.extract_boundary(turf_mask)

        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 3. Visualize
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
        """
        Processes an input video file stream frame-by-frame.
        """
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
                    cv2.imshow("Pitch Segmentation - Phase 1", rendered_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] User interrupted video playback.")
                        break

                if frame_count % 30 == 0 or frame_count == total_frames:
                    print(
                        f"[PROGRESS] Frame {frame_count}/{total_frames} | "
                        f"Mode: {meta['mode_used']} | FPS: {meta['fps']:.1f}"
                    )
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        print(f"[SUCCESS] Video processing complete. Processed {frame_count} frames.")


def main():
    """
    Entrypoint for Pitch Segmentation Pipeline.
    Configurable paths and variables.
    """
    # ------------------ CONFIGURATION ------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    # Default paths
    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "pitch_seg_yolov8.pt")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pitch_segmentation_output.mp4")
    CONFIDENCE_THRESHOLD = 0.50

    print("==================================================")
    print("      FOOTBALL PITCH SEGMENTATION - PHASE 1       ")
    print("==================================================")

    # Ensure required folders exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Pipeline
    pipeline = PitchSegmentationPipeline(
        model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD
    )

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH, display=False)
    else:
        print(f"\n[NOTICE] Input video file not found at: {VIDEO_PATH}")
        print("Folder structure initialized:")
        print(f" -> Input Video Path: {VIDEO_PATH}")
        print(f" -> Custom Weights Path: {MODEL_PATH}")
        print(f" -> Output Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
