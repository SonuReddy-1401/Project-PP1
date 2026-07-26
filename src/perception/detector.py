import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import supervision as sv
from ultralytics import YOLO

class FootballDetector:
    """
    Object detector wrapper for football entities (Players, Referees, Ball).
    Optimized for high-resolution wide-angle tactical broadcast footage.
    Uses Ultralytics YOLOv8 models.
    """
    def __init__(self, model_path: str = "yolov8m.pt", conf_threshold: float = 0.15, imgsz: int = 1280, iou_threshold: float = 0.45):
        """
        Initializes YOLO model.
        COCO classes:
          0: person (players/referees)
          32: sports ball (ball)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        
        # COCO class mapping
        self.PERSON_CLASS_ID = 0
        self.BALL_CLASS_ID = 32

    def detect_frame(self, frame: np.ndarray) -> sv.Detections:
        """
        Runs object detection on a single video frame with high-resolution scaling.
        Returns a supervision Detections object containing bboxes, confidences, and class_ids.
        """
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            imgsz=self.imgsz, 
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        
        if len(detections) == 0 or detections.class_id is None:
            return sv.Detections.empty()

        # Filter only person and ball classes
        target_indices = np.isin(detections.class_id, [self.PERSON_CLASS_ID, self.BALL_CLASS_ID])
        filtered_detections = detections[target_indices]
        
        return filtered_detections

    def separate_detections(self, detections: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
        """
        Separates detections into (players_and_refs, ball).
        """
        if len(detections) == 0 or detections.class_id is None:
            return sv.Detections.empty(), sv.Detections.empty()

        players_mask = detections.class_id == self.PERSON_CLASS_ID
        ball_mask = detections.class_id == self.BALL_CLASS_ID
        
        return detections[players_mask], detections[ball_mask]
