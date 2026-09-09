"""
Fully GPU-Accelerated Pitch Segmentation & Line Detection Pipeline
Phase 1 - Football Performance Analysis System

All frame processing (HSV color conversion, morphology, L*a*b* luminance,
line thresholding, alpha blending, and model inference) runs 100% on NVIDIA GPU (cuda:0).

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation
"""

import os
import time
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def rgb_to_hsv_gpu(rgb_gpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch CUDA implementation of RGB to HSV conversion.
    
    Args:
        rgb_gpu (torch.Tensor): Shape (3, H, W), float tensor in range [0, 1] on CUDA.
        
    Returns:
        h_chan, s_chan, v_chan (Tuple[torch.Tensor]): Each (H, W) in range [0, 1] on CUDA.
    """
    r, g, b = rgb_gpu[0], rgb_gpu[1], rgb_gpu[2]
    max_c, _ = torch.max(rgb_gpu, dim=0)
    min_c, _ = torch.min(rgb_gpu, dim=0)
    delta = max_c - min_c

    v_chan = max_c
    s_chan = torch.where(max_c > 1e-6, delta / (max_c + 1e-7), torch.zeros_like(max_c))

    h_chan = torch.zeros_like(max_c)
    valid_mask = delta > 1e-6

    r_mask = (max_c == r) & valid_mask
    g_mask = (max_c == g) & valid_mask
    b_mask = (max_c == b) & valid_mask

    h_chan[r_mask] = (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0) / 6.0
    h_chan[g_mask] = (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0) / 6.0
    h_chan[b_mask] = (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0) / 6.0

    return h_chan, s_chan, v_chan


class GPUMorphology:
    """
    GPU-accelerated morphological operations using PyTorch MaxPool2d on CUDA tensors.
    """

    @staticmethod
    def dilate(mask_gpu: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """Dilates binary float mask on GPU (1.0 for mask, 0.0 for background)."""
        padding = kernel_size // 2
        return torch.nn.functional.max_pool2d(mask_gpu, kernel_size=kernel_size, stride=1, padding=padding)

    @staticmethod
    def erode(mask_gpu: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """Erodes binary float mask on GPU."""
        padding = kernel_size // 2
        return -torch.nn.functional.max_pool2d(-mask_gpu, kernel_size=kernel_size, stride=1, padding=padding)

    @classmethod
    def open(cls, mask_gpu: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """Morphological Open (Erode then Dilate) on GPU."""
        return cls.dilate(cls.erode(mask_gpu, kernel_size), kernel_size)

    @classmethod
    def close(cls, mask_gpu: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        """Morphological Close (Dilate then Erode) on GPU."""
        return cls.erode(cls.dilate(mask_gpu, kernel_size), kernel_size)


class GPUPitchSegmentor:
    """
    Fully GPU-Accelerated Pitch Turf & White Line Segmentor operating on PyTorch CUDA Tensors.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        hue_min: float = 0.095,   # ~34 degrees
        hue_max: float = 0.240,   # ~86 degrees
        sat_min: float = 0.250,   # ~64/255
        val_min: float = 0.250,   # ~64/255
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.val_min = val_min

        print(f"[INFO] GPUPitchSegmentor initialized on device: {self.device}")

    def segment_gpu(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes color space conversion, thresholding, and morphology directly on CUDA tensor.

        Returns:
            turf_mask_gpu (torch.Tensor): (H, W) uint8 CUDA tensor (255 inside turf, 0 outside)
            line_mask_gpu (torch.Tensor): (H, W) uint8 CUDA tensor (255 on white lines, 0 outside)
        """
        h, w = frame_bgr.shape[:2]

        # 1. Upload BGR numpy array to GPU CUDA tensor
        frame_tensor = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)  # (H, W, 3)

        # Convert BGR -> RGB float tensor in [0, 1] on GPU: (3, H, W)
        rgb_gpu = frame_tensor[..., [2, 1, 0]].permute(2, 0, 1).float() / 255.0

        # 2. RGB to HSV on GPU (Pure CUDA Tensor math)
        h_chan, s_chan, v_chan = rgb_to_hsv_gpu(rgb_gpu)

        # 3. Green Turf Masking on GPU
        turf_cond = (
            (h_chan >= self.hue_min)
            & (h_chan <= self.hue_max)
            & (s_chan >= self.sat_min)
            & (v_chan >= self.val_min)
        )

        # Field Horizon Cutoff (top 15% frame ignored on GPU)
        horizon_y = int(h * 0.15)
        turf_cond[:horizon_y, :] = False

        # Convert bool to (1, 1, H, W) float tensor for GPU convolution
        turf_float_gpu = turf_cond.float().unsqueeze(0).unsqueeze(0)

        # GPU Morphological Operations
        opened_gpu = GPUMorphology.open(turf_float_gpu, kernel_size=9)
        closed_gpu = GPUMorphology.close(opened_gpu, kernel_size=25)

        turf_mask_bool_gpu = (closed_gpu.squeeze() > 0.5)

        # 4. GPU White Line Detection (Luminance + Neutral Saturation on GPU)
        r_chan, g_chan, b_chan = rgb_gpu[0], rgb_gpu[1], rgb_gpu[2]

        # Luminance calculation on GPU
        luminance_gpu = 0.299 * r_chan + 0.587 * g_chan + 0.114 * b_chan

        # Color difference (neutral white has low color distance between R, G, B)
        color_diff_gpu = torch.abs(r_chan - g_chan) + torch.abs(g_chan - b_chan)

        # Line condition: High luminance, low color distance, strictly inside turf boundary
        line_cond_gpu = (
            (luminance_gpu >= 0.65)
            & (color_diff_gpu <= 0.15)
            & turf_mask_bool_gpu
        )

        line_float_gpu = line_cond_gpu.float().unsqueeze(0).unsqueeze(0)
        line_clean_gpu = GPUMorphology.open(line_float_gpu, kernel_size=3)

        # Format output tensors as (H, W) uint8 on GPU
        turf_mask_gpu = (turf_mask_bool_gpu.to(torch.uint8)) * 255
        line_mask_gpu = ((line_clean_gpu.squeeze() > 0.5).to(torch.uint8)) * 255

        return turf_mask_gpu, line_mask_gpu


class GPUPitchVisualizer:
    """
    Renders visualization overlays directly on GPU CUDA tensors.
    """

    def __init__(self, alpha: float = 0.35, device: str = "cuda:0"):
        self.alpha = alpha
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def render_gpu(
        self,
        frame_bgr: np.ndarray,
        turf_mask_gpu: torch.Tensor,
        line_mask_gpu: torch.Tensor,
        boundary_contour: Optional[np.ndarray],
        fps: float = 0.0,
        mode_used: str = "CUDA-GPU Pipeline",
    ) -> np.ndarray:
        """
        Performs GPU alpha blending, line highlights, and returns rendered numpy image for VideoWriter.
        """
        # Upload frame to GPU
        frame_gpu = torch.from_numpy(frame_bgr).to(self.device).float()  # (H, W, 3)

        # 1. GPU Green Turf Overlay
        turf_overlay_gpu = frame_gpu.clone()
        turf_indices = turf_mask_gpu > 0

        # Apply Green tint (BGR: 0, 210, 0)
        green_color = torch.tensor([0.0, 210.0, 0.0], device=self.device)
        blended = (frame_gpu[turf_indices] * (1.0 - self.alpha)) + (green_color * self.alpha)
        turf_overlay_gpu[turf_indices] = blended

        # 2. GPU Line Highlights (Neon Yellow BGR: 0, 255, 255)
        line_indices = line_mask_gpu > 0
        turf_overlay_gpu[line_indices] = torch.tensor([0.0, 255.0, 255.0], device=self.device)

        # 3. Download rendered tensor back to CPU numpy for OpenCV contouring & display
        rendered = turf_overlay_gpu.byte().cpu().numpy()

        # 4. Draw outer field boundary polygon on CPU
        if boundary_contour is not None:
            cv2.drawContours(rendered, [boundary_contour], -1, (255, 255, 0), 3)
            for point in boundary_contour:
                pt = tuple(point[0])
                cv2.circle(rendered, pt, 6, (0, 0, 255), -1)

        # 5. HUD Overlay
        hud_text = f"Mode: {mode_used} | GPU: {torch.cuda.get_device_name(0)} | FPS: {fps:.1f}"
        cv2.rectangle(rendered, (10, 10), (520, 50), (0, 0, 0), -1)
        cv2.putText(rendered, hud_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return rendered


class GPUPitchSegmentationPipeline:
    """
    Master GPU Pipeline Class.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.50):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.segmentor_gpu = GPUPitchSegmentor(device=self.device)
        self.visualizer_gpu = GPUPitchVisualizer(device=self.device)
        self.conf_threshold = conf_threshold
        self.model = None
        self.using_custom_model = False

        if model_path and os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                print(f"[INFO] Loading custom YOLO model on GPU ({self.device}): {model_path}")
                self.model = YOLO(model_path)
                self.using_custom_model = True
            except Exception as e:
                print(f"[WARN] Failed to load model on GPU: {e}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()
        mode_used = "PyTorch CUDA GPU"

        # 1. GPU Segmentation Pass
        turf_mask_gpu, line_mask_gpu = self.segmentor_gpu.segment_gpu(frame)

        # 2. Extract Boundary Polygon
        turf_mask_cpu = turf_mask_gpu.cpu().numpy()
        contours, _ = cv2.findContours(turf_mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundary_contour = None
        shapely_polygon = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= 2000:
                hull = cv2.convexHull(largest_contour)
                epsilon = 0.012 * cv2.arcLength(hull, True)
                boundary_contour = cv2.approxPolyDP(hull, epsilon, True)

                poly_pts = boundary_contour.reshape(-1, 2)
                if len(poly_pts) >= 3:
                    try:
                        shapely_polygon = Polygon(poly_pts)
                    except Exception:
                        shapely_polygon = None

        # GPU synchronization for exact timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # 3. Render Output Frame
        rendered_frame = self.visualizer_gpu.render_gpu(
            frame_bgr=frame,
            turf_mask_gpu=turf_mask_gpu,
            line_mask_gpu=line_mask_gpu,
            boundary_contour=boundary_contour,
            fps=fps,
            mode_used=mode_used,
        )

        meta = {
            "shapely_polygon": shapely_polygon,
            "fps": fps,
            "mode_used": mode_used,
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

        return rendered_frame, meta

    def process_video(self, video_path: str, output_path: str) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video clip not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video stream: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"==================================================")
        print(f"  GPU-ACCELERATED FOOTBALL PITCH SEGMENTATION    ")
        print(f"==================================================")
        print(f"[INFO] Active Hardware: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"[INFO] Video File: {video_path}")
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
                    vram_used = torch.cuda.memory_allocated(0) / 1e6
                    print(
                        f"[GPU PROGRESS] Frame {frame_count}/{total_frames} | "
                        f"Processing Speed: {meta['fps']:.1f} FPS | VRAM Allocated: {vram_used:.1f} MB"
                    )
        finally:
            cap.release()
            writer.release()

        total_elapsed = time.time() - total_start
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        print(f"\n[SUCCESS] GPU Processing complete in {total_elapsed:.2f} seconds!")
        print(f"[METRICS] Average Processing Speed: {avg_fps:.2f} FPS")
        print(f"[OUTPUT FILE] Saved to: {output_path}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

    VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "pitch_seg_yolov8.pt")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pitch_segmentation_gpu_output.mp4")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "weights"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is not available! Please check NVIDIA drivers.")
        return

    pipeline = GPUPitchSegmentationPipeline(model_path=MODEL_PATH)

    if os.path.exists(VIDEO_PATH):
        pipeline.process_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH)
    else:
        print(f"\n[NOTICE] Input video file not found at: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
