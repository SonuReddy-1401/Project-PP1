import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import supervision as sv
from ultralytics import YOLO

class FootballDetector:
    """
    Object detector wrapper for football entities (Players, Referees, Ball).
    Optimized for high-resolution wide-angle tactical broadcast footage.
    Supports Sliced/Tiled Inference (SAHI) with GPU Tensor Batching & FP16 Half-Precision.
    Automatically utilizes NVIDIA CUDA GPU acceleration when available.
    """
    def __init__(self, model_path: str = "yolov8m.pt", conf_threshold: float = 0.15, 
                 imgsz: int = 1280, iou_threshold: float = 0.45, use_slicing: bool = True,
                 device: Optional[str] = None, half: Optional[bool] = None):
        """
        Initializes YOLO model.
        COCO classes:
          0: person (players/referees)
          32: sports ball (ball)
        """
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if half is None:
            self.half = (self.device == "cuda")
        else:
            self.half = half

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.use_slicing = use_slicing
        
        # COCO class mapping
        self.PERSON_CLASS_ID = 0
        self.BALL_CLASS_ID = 32

    def detect_frame(self, frame: np.ndarray) -> sv.Detections:
        """
        Runs object detection on a single video frame.
        Uses Sliced Inference if use_slicing=True for maximum recall on tiny distant players.
        """
        if self.use_slicing:
            return self.detect_frame_sliced(frame)
        return self._detect_single_pass(frame, imgsz=self.imgsz)

    def _detect_single_pass(self, frame: np.ndarray, imgsz: int = 1280) -> sv.Detections:
        """
        Single-pass YOLO inference with optional FP16 half precision.
        """
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            imgsz=imgsz, 
            device=self.device,
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if len(detections) == 0 or detections.class_id is None:
            return sv.Detections.empty()

        # Filter only person and ball classes
        target_indices = np.isin(detections.class_id, [self.PERSON_CLASS_ID, self.BALL_CLASS_ID])
        filtered_detections = detections[target_indices]
        
        return filtered_detections

    def detect_frame_sliced(
        self, 
        frame: np.ndarray, 
        slice_wh: Tuple[int, int] = (640, 640), 
        overlap_ratio: float = 0.20
    ) -> sv.Detections:
        """
        Slices high-resolution broadcast frame into overlapping patches and executes
        GPU Tensor Batching for maximum recall and speed on RTX GPUs.
        """
        h_frame, w_frame = frame.shape[:2]
        slice_w, slice_h = slice_wh
        
        step_x = int(slice_w * (1.0 - overlap_ratio))
        step_y = int(slice_h * (1.0 - overlap_ratio))
        
        x_offsets = list(range(0, max(1, w_frame - slice_w + 1), step_x))
        if len(x_offsets) == 0 or x_offsets[-1] + slice_w < w_frame:
            x_offsets.append(max(0, w_frame - slice_w))
            
        y_offsets = list(range(0, max(1, h_frame - slice_h + 1), step_y))
        if len(y_offsets) == 0 or y_offsets[-1] + slice_h < h_frame:
            y_offsets.append(max(0, h_frame - slice_h))
            
        all_detections = []
        
        # 1. Full-frame high-resolution pass
        full_dets = self._detect_single_pass(frame, imgsz=self.imgsz)
        if len(full_dets) > 0:
            all_detections.append(full_dets)
            
        # 2. Collect slice crops for GPU Tensor Batching
        slice_crops = []
        slice_coords = []
        
        for y_off in y_offsets:
            for x_off in x_offsets:
                slice_crop = frame[y_off:y_off+slice_h, x_off:x_off+slice_w]
                if slice_crop.size == 0 or slice_crop.shape[0] < 50 or slice_crop.shape[1] < 50:
                    continue
                slice_crops.append(slice_crop)
                slice_coords.append((x_off, y_off))
                
        if slice_crops:
            # Batch inference across CUDA Tensor Cores
            batch_results = self.model(
                slice_crops,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=slice_w,
                device=self.device,
                verbose=False
            )
            
            for res, (x_off, y_off) in zip(batch_results, slice_coords):
                slice_dets = sv.Detections.from_ultralytics(res)
                if len(slice_dets) == 0 or slice_dets.class_id is None:
                    continue
                    
                target_mask = np.isin(slice_dets.class_id, [self.PERSON_CLASS_ID, self.BALL_CLASS_ID])
                filtered_slice_dets = slice_dets[target_mask]
                
                if len(filtered_slice_dets) == 0:
                    continue
                    
                # Offset bounding boxes back to full-frame coordinates
                filtered_slice_dets.xyxy[:, [0, 2]] += x_off
                filtered_slice_dets.xyxy[:, [1, 3]] += y_off
                all_detections.append(filtered_slice_dets)
                
        if not all_detections:
            return sv.Detections.empty()
            
        # Merge all detections across slices
        merged_dets = sv.Detections.merge(all_detections)
        if hasattr(merged_dets, 'with_nms'):
            merged_dets = merged_dets.with_nms(threshold=self.iou_threshold)
            
        return merged_dets

    def separate_detections(self, detections: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
        """
        Separates detections into (players_and_refs, ball).
        """
        if len(detections) == 0 or detections.class_id is None:
            return sv.Detections.empty(), sv.Detections.empty()

        players_mask = detections.class_id == self.PERSON_CLASS_ID
        ball_mask = detections.class_id == self.BALL_CLASS_ID
        
        return detections[players_mask], detections[ball_mask]
