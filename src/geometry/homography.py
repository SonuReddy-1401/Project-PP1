import cv2
import numpy as np
from typing import List, Tuple, Optional, Any

class HomographyTransformer:
    """
    Computes Perspective Homography Matrix H mapping screen pixels (x, y)
    to real-world 2D pitch metric coordinates (X_meters, Y_meters).
    """
    def __init__(self, src_pixels: Optional[np.ndarray] = None, dst_meters: Optional[np.ndarray] = None):
        """
        src_pixels: 4 or more (x, y) screen pixel coordinates.
        dst_meters: 4 or more corresponding (X, Y) meter coordinates on tactical pitch.
        """
        self.H: Optional[np.ndarray] = None
        self.inv_H: Optional[np.ndarray] = None
        
        if src_pixels is not None and dst_meters is not None:
            self.update_homography(src_pixels, dst_meters)

    def update_homography(self, src_pixels: np.ndarray, dst_meters: np.ndarray, smoothing_alpha: float = 0.85):
        """
        Computes 3x3 Homography matrix using OpenCV cv2.findHomography with temporal EMA smoothing.
        Prevents 2D pitch map player dots from jumping when camera pans.
        """
        src_pts = np.float32(src_pixels).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_meters).reshape(-1, 1, 2)
        
        new_H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if new_H is not None:
            if self.H is None:
                self.H = new_H
            else:
                self.H = smoothing_alpha * self.H + (1.0 - smoothing_alpha) * new_H
            self.inv_H = np.linalg.inv(self.H)

    def pixel_to_pitch(self, pixel_pos: np.ndarray) -> np.ndarray:
        """
        Transforms a single pixel point (x, y) or batch of points (N, 2)
        to pitch meter coordinates (X, Y).
        """
        if self.H is None:
            # Fallback identity scaling if homography not calibrated
            return pixel_pos * 0.05
            
        pts = np.float32(pixel_pos).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2)

    def pitch_to_pixel(self, pitch_pos: np.ndarray) -> np.ndarray:
        """
        Transforms tactical pitch meter coordinates (X, Y) back to screen pixel coordinates.
        """
        if self.inv_H is None:
            return pitch_pos / 0.05
            
        pts = np.float32(pitch_pos).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.inv_H)
        return transformed.reshape(-1, 2)

    def is_inside_pitch(self, pitch_pos: Any, margin: float = 2.0, 
                        pitch_length: float = 105.0, pitch_width: float = 68.0) -> bool:
        """
        Returns boolean indicating whether pitch coordinates (X, Y) lie inside valid play bounds.
        Filters out sideline staff, coaches, substitute players, and camera crew.
        """
        try:
            arr = np.asarray(pitch_pos, dtype=np.float32).flatten()
            if len(arr) < 2:
                return False
            x, y = float(arr[0]), float(arr[1])
            if np.isnan(x) or np.isnan(y):
                return False
            return (-margin <= x <= pitch_length + margin) and (-margin <= y <= pitch_width + margin)
        except Exception:
            return False
