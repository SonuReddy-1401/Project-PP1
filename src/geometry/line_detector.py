import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class PitchLineDetector:
    """
    Detects white pitch line markings and keypoint intersections in broadcast video frames.
    Provides diagnostic visual overlay modes for verifying pitch line detection accuracy.
    Dynamically updates Homography Matrix H on every frame to handle camera panning, tilting, and zooming.
    """
    def __init__(self, ref_width: int = 1280, ref_height: int = 720):
        self.ref_w = ref_width
        self.ref_h = ref_height

    def extract_white_line_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts white pitch line pixels while filtering out green grass.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White color mask (Low saturation, High value/brightness)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        
        # Green grass mask to constrain line search to the pitch
        grass_mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([90, 255, 255]))
        
        # Morphological dilation of grass mask to include lines inside grass area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated_grass = cv2.dilate(grass_mask, kernel, iterations=2)
        
        line_mask = cv2.bitwise_and(white_mask, white_mask, mask=dilated_grass)
        return line_mask

    def detect_pitch_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects structural pitch line intersections in the broadcast frame.
        Returns: src_pixels keypoints array or None if insufficient keypoints detected.
        """
        line_mask = self.extract_white_line_mask(frame)
        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)
        
        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=15)
        if lines is None or len(lines) < 4:
            return None

        # Filter horizontal vs vertical line segments
        horiz_lines = []
        vert_lines = []
        for line in lines:
            line_pts = line[0] if line.ndim > 1 else line
            x1, y1, x2, y2 = map(int, line_pts)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle < 35.0 or angle > 145.0:
                horiz_lines.append((x1, y1, x2, y2))
            elif 55.0 < angle < 125.0:
                vert_lines.append((x1, y1, x2, y2))

        if not horiz_lines or not vert_lines:
            return None

        # Calculate line intersections
        intersections = []
        for h_line in horiz_lines[:20]:
            hx1, hy1, hx2, hy2 = h_line
            for v_line in vert_lines[:20]:
                vx1, vy1, vx2, vy2 = v_line
                
                A1, B1, C1 = hy2 - hy1, hx1 - hx2, (hy2 - hy1) * hx1 + (hx1 - hx2) * hy1
                A2, B2, C2 = vy2 - vy1, vx1 - vx2, (vy2 - vy1) * vx1 + (vx1 - vx2) * vy1
                
                det = A1 * B2 - A2 * B1
                if np.abs(det) > 1e-3:
                    ix = (B2 * C1 - B1 * C2) / det
                    iy = (A1 * C2 - A2 * C1) / det
                    
                    if 0 <= ix < frame.shape[1] and 0 <= iy < frame.shape[0]:
                        intersections.append((ix, iy))

        if len(intersections) < 4:
            return None

        # Cluster nearby intersections
        pts = np.float32(intersections)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        n_clusters = min(8, len(pts))
        _, _, centers = cv2.kmeans(pts, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers

    def draw_pitch_line_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders high-contrast, color-coded pitch line overlays directly on the broadcast video frame.
          - Magenta (255, 0, 255): Horizontal Touchlines & Penalty Box Boundaries
          - Cyan (255, 255, 0): Vertical Goal Lines & Penalty Area Lines
          - Yellow (0, 255, 255): Other Field Segments
          - Bright Green Circles (0, 255, 0): Structural Intersection Keypoints
        """
        annotated = frame.copy()
        line_mask = self.extract_white_line_mask(frame)
        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
        
        line_count = 0
        if lines is not None:
            for line in lines:
                line_pts = line[0] if line.ndim > 1 else line
                x1, y1, x2, y2 = map(int, line_pts)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                
                # Color code by line orientation
                if angle < 35.0 or angle > 145.0:
                    color = (255, 0, 255)  # Magenta for Touchlines / Horizontal
                elif 55.0 < angle < 125.0:
                    color = (255, 255, 0)  # Cyan for Goal Lines / Vertical
                else:
                    color = (0, 255, 255)  # Yellow for Diagonals / Arcs
                    
                cv2.line(annotated, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                cv2.circle(annotated, (x1, y1), 3, (255, 255, 255), -1)
                cv2.circle(annotated, (x2, y2), 3, (255, 255, 255), -1)
                line_count += 1

        # Detect structural keypoint intersections
        keypoints = self.detect_pitch_keypoints(frame)
        kp_count = 0
        if keypoints is not None:
            for kp in keypoints:
                kx, ky = int(kp[0]), int(kp[1])
                cv2.circle(annotated, (kx, ky), 9, (0, 0, 0), -1)
                cv2.circle(annotated, (kx, ky), 7, (0, 255, 0), -1)  # Green Keypoint Dot
                cv2.circle(annotated, (kx, ky), 8, (255, 255, 255), 1, cv2.LINE_AA)
                kp_count += 1

        # Render HUD Banner at top of frame
        banner_text = f"[PITCH LINE DETECTOR DEBUG] Detected Line Segments: {line_count} | Field Keypoint Intersections: {kp_count}"
        cv2.rectangle(annotated, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(annotated, banner_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        
        return annotated

    def update_homography_dynamically(
        self, 
        frame: np.ndarray, 
        homography_transformer: Any
    ) -> bool:
        """
        Detects pitch lines in the frame and recalibrates homography transformer dynamically.
        Returns True if dynamic recalibration succeeded, False if using previous frame matrix.
        """
        keypoints = self.detect_pitch_keypoints(frame)
        if keypoints is None or len(keypoints) < 4:
            return False

        sorted_kps = sorted(keypoints, key=lambda p: (p[1], p[0]))
        src_pts = np.float32(sorted_kps[:4])
        
        dst_m = np.array([
            [15.0, 13.84],   # Left Penalty Box Top-Left
            [90.0, 13.84],   # Right Penalty Box Top-Right
            [90.0, 54.16],   # Right Penalty Box Bottom-Right
            [15.0, 54.16]    # Left Penalty Box Bottom-Left
        ], dtype=np.float32)

        try:
            homography_transformer.update_homography(src_pts, dst_m)
            return True
        except Exception:
            return False
