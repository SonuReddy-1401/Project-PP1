import cv2
import numpy as np
from typing import List, Tuple, Optional

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

    def update_homography(self, src_pixels: np.ndarray, dst_meters: np.ndarray):
        """
        Computes 3x3 Homography matrix using OpenCV cv2.findHomography.
        """
        src_pts = np.float32(src_pixels).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_meters).reshape(-1, 1, 2)
        
        self.H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if self.H is not None:
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

    def is_inside_pitch(self, pitch_pos: np.ndarray, margin: float = 2.0, 
                        pitch_length: float = 105.0, pitch_width: float = 68.0) -> np.ndarray:
        """
        Returns boolean mask indicating whether pitch coordinates (X, Y) lie inside valid play bounds.
        Filters out sideline staff, coaches, substitute players, and camera crew.
        """
        pts = np.atleast_2d(pitch_pos)
        x = pts[:, 0]
        y = pts[:, 1]
        
        valid_x = (x >= -margin) & (x <= pitch_length + margin)
        valid_y = (y >= -margin) & (y <= pitch_width + margin)
        is_inside = valid_x & valid_y
        
        if pitch_pos.ndim == 1:
            return bool(is_inside[0])
        return is_inside
