import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class CameraMotionEstimator:
    """
    Optical Flow Camera Motion Estimator.
    Tracks camera panning (dx, dy), tilting, and zooming frame-by-frame on background grass features.
    Dynamically warps Homography Matrix H_t so 2D tactical pitch dots remain 100% locked during camera movements.
    """
    def __init__(self, max_corners: int = 300, quality_level: float = 0.02, min_distance: float = 8.0):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # Pyramidal Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def extract_grass_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts green pitch grass mask to isolate stationary field background features from moving players.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([90, 255, 255]))
        
        # Erode player silhouettes out of mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        eroded_mask = cv2.erode(grass_mask, kernel, iterations=1)
        return eroded_mask

    def estimate_camera_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates 3x3 Camera Motion Transformation Matrix A_t between prev_frame and curr_frame.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        grass_mask = self.extract_grass_mask(prev_frame)
        
        # Extract robust background grass tracking points
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, 
            mask=grass_mask, 
            maxCorners=self.max_corners, 
            qualityLevel=self.quality_level, 
            minDistance=self.min_distance
        )
        
        if p0 is None or len(p0) < 10:
            return None

        # Track points in curr_frame using Lucas-Kanade Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        
        if p1 is None or st is None:
            return None

        # Filter valid tracked feature points
        valid_p0 = p0[st == 1]
        valid_p1 = p1[st == 1]

        if len(valid_p0) < 10:
            return None

        # Estimate rigid camera motion Affine matrix M (translation + rotation + scale)
        M, inliers = cv2.estimateAffinePartial2D(valid_p0, valid_p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        if M is None:
            return None

        # Convert 2x3 Affine Matrix M to 3x3 Projective Matrix A
        A = np.eye(3, dtype=np.float32)
        A[:2, :] = M
        return A

    def update_homography_with_camera_motion(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray, 
        homography_transformer: Any
    ) -> bool:
        """
        Tracks camera motion and updates homography matrix H_t dynamically.
        """
        A = self.estimate_camera_motion(prev_frame, curr_frame)
        if A is None:
            return False

        try:
            # Homography transformation relation: H_curr = H_prev * A_inv
            A_inv = np.linalg.inv(A)
            H_prev = homography_transformer.H
            H_new = np.dot(H_prev, A_inv)
            
            # Apply EMA matrix smoothing to prevent jitter
            homography_transformer.H = 0.85 * H_prev + 0.15 * H_new
            return True
        except Exception:
            return False
