import cv2
import numpy as np

class PitchTransformer:
    """
    Roboflow Sports (football-ai.ipynb) 32-Keypoint FIFA Pitch Homography ViewTransformer (105m x 68m).
    """
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0):
        self.pitch_length = pitch_length_m
        self.pitch_width = pitch_width_m

        # Default 4 corners of 1080p broadcast camera view mapped to FIFA metric pitch (105m x 68m)
        self.src_points = np.float32([
            [300, 250],   # Top-Left
            [1620, 250],  # Top-Right
            [1850, 950],  # Bottom-Right
            [70, 950]     # Bottom-Left
        ])

        self.dst_points = np.float32([
            [0.0, 0.0],
            [105.0, 0.0],
            [105.0, 68.0],
            [0.0, 68.0]
        ])

        self.homography_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def transform_point(self, point_px: tuple) -> tuple:
        """
        Transforms a pixel coordinate (x, y) into metric pitch coordinates (x_m, y_m).
        """
        pt = np.array([[[float(point_px[0]), float(point_px[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        x_m = float(transformed[0][0][0])
        y_m = float(transformed[0][0][1])

        # Clamp to pitch metric boundaries
        x_m = max(0.0, min(105.0, x_m))
        y_m = max(0.0, min(68.0, y_m))
        return (x_m, y_m)
