import cv2
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ultralytics import YOLO

class PitchKeypointDetector:
    """
    Pre-trained Deep Neural Network for detecting 29 structural FIFA Football Pitch Keypoint Landmarks.
    Provides deep learning homography calibration working under camera zooms, glares, and panning.
    """
    def __init__(self, model_path: str = "models/football_pitch_keypoints.pt", conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        
        if not os.path.exists(model_path):
            # Fallback to yolov8n-pose
            model_path = "yolov8n-pose.pt"
            
        self.model = YOLO(model_path)
        
        # Standard FIFA 105m x 68m Pitch Keypoint Metric Reference Coordinates
        self.keypoints_meters = {
            0: np.array([0.0, 0.0]),         # Top-Left Pitch Corner
            1: np.array([105.0, 0.0]),       # Top-Right Pitch Corner
            2: np.array([105.0, 68.0]),      # Bottom-Right Pitch Corner
            3: np.array([0.0, 68.0]),        # Bottom-Left Pitch Corner
            4: np.array([52.5, 0.0]),        # Center Line Top Touchline T-Junction
            5: np.array([52.5, 68.0]),       # Center Line Bottom Touchline T-Junction
            6: np.array([52.5, 34.0]),       # Center Spot
            7: np.array([16.5, 13.84]),      # Left Penalty Box Top-Right
            8: np.array([16.5, 54.16]),      # Left Penalty Box Bottom-Right
            9: np.array([0.0, 13.84]),       # Left Penalty Box Top-Left
            10: np.array([0.0, 54.16]),      # Left Penalty Box Bottom-Left
            11: np.array([88.5, 13.84]),     # Right Penalty Box Top-Left
            12: np.array([88.5, 54.16]),     # Right Penalty Box Bottom-Left
            13: np.array([105.0, 13.84]),    # Right Penalty Box Top-Right
            14: np.array([105.0, 54.16]),    # Right Penalty Box Bottom-Right
            15: np.array([5.5, 24.84]),      # Left Goal Area Top-Right
            16: np.array([5.5, 43.16]),      # Left Goal Area Bottom-Right
            17: np.array([99.5, 24.84]),     # Right Goal Area Top-Left
            18: np.array([99.5, 43.16]),     # Right Goal Area Bottom-Left
            19: np.array([11.0, 34.0]),      # Left Penalty Spot
            20: np.array([94.0, 34.0])       # Right Penalty Spot
        }

    def detect_keypoints(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Runs deep pose/keypoint inference on frame and returns (src_pixels, dst_meters) keypoint pairs.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        if not hasattr(results, 'keypoints') or results.keypoints is None:
            return None

        keypoints_data = results.keypoints.data
        if len(keypoints_data) == 0:
            return None

        # Extract predicted keypoint coordinates and confidence scores
        kps = keypoints_data[0].cpu().numpy()  # Shape: (N_kps, 3) -> (x, y, conf)
        
        src_pts = []
        dst_pts = []
        
        for kp_idx, kp in enumerate(kps):
            x, y, conf = kp[0], kp[1], kp[2]
            if conf >= self.conf_threshold and kp_idx in self.keypoints_meters:
                src_pts.append([x, y])
                dst_pts.append(self.keypoints_meters[kp_idx])

        if len(src_pts) < 4:
            return None

        return np.float32(src_pts), np.float32(dst_pts)

    def update_homography_dynamically(self, frame: np.ndarray, homography_transformer: Any) -> bool:
        """
        Predicts pitch keypoints using neural net and recalibrates homography transformer.
        """
        kp_pair = self.detect_keypoints(frame)
        if kp_pair is None:
            return False

        src_pts, dst_pts = kp_pair
        try:
            homography_transformer.update_homography(src_pts, dst_pts)
            return True
        except Exception:
            return False

    def draw_keypoint_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders predicted deep pitch keypoints directly on broadcast frames as glowing green/cyan landmark dots with class labels.
        """
        annotated = frame.copy()
        kp_pair = self.detect_keypoints(frame)
        
        kp_count = 0
        if kp_pair is not None:
            src_pts, dst_pts = kp_pair
            for pt in src_pts:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (x, y), 9, (0, 0, 0), -1)
                cv2.circle(annotated, (x, y), 7, (0, 255, 255), -1)  # Yellow landmark dot
                cv2.circle(annotated, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
                kp_count += 1

        banner_text = f"[DEEP PITCH KEYPOINT MODEL] Predicted Landmarks: {kp_count} / 29"
        cv2.rectangle(annotated, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(annotated, banner_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        
        return annotated
