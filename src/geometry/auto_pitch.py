import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class AutoPitchCalibrator:
    """
    Automatic Green Pitch Grass Contour & Field Perspective Calibrator.
    Extracts visible pitch boundary corners from raw video frames and computes initial homography H_0
    matching standard FIFA 105m x 68m pitch metric coordinates.
    """
    def __init__(self):
        pass

    def extract_grass_polygon_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the 4 outer corner points [top_left, top_right, bottom_right, bottom_left]
        of the visible playing field from green grass HSV segmentation.
        """
        h_frame, w_frame = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green grass mask
        grass_mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([90, 255, 255]))
        
        # Morphological closing & dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Take largest grass contour
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) < (h_frame * w_frame * 0.15):
            return None

        # Convex hull of pitch region
        hull = cv2.convexHull(largest_c)
        
        # Find 4 extremal corners of hull: (sum and diff of x, y)
        pts = hull.reshape(-1, 2)
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)

        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]

        src_corners = np.float32([top_left, top_right, bottom_right, bottom_left])
        return src_corners

    def calibrate_initial_homography(self, frame: np.ndarray, homography_transformer: Any) -> bool:
        """
        Calibrates initial homography matrix H_0 based on automatically detected visible pitch corners.
        """
        src_corners = self.extract_grass_polygon_corners(frame)
        if src_corners is None:
            return False

        # FIFA 105m x 68m Tactical Pitch Destination Metric Coordinates
        dst_m = np.array([
            [10.0, 0.0],    # Top-Left Metric Boundary
            [95.0, 0.0],    # Top-Right Metric Boundary
            [95.0, 68.0],   # Bottom-Right Metric Boundary
            [10.0, 68.0]    # Bottom-Left Metric Boundary
        ], dtype=np.float32)

        try:
            homography_transformer.update_homography(src_corners, dst_m)
            print(f"[AUTO PITCH CALIBRATION] Pitch bounds detected: Top-Left {src_corners[0]} -> Bottom-Right {src_corners[2]}")
            return True
        except Exception:
            return False
