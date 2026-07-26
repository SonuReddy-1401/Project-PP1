import cv2
import numpy as np
from typing import Tuple

class TacticalPitchTemplate:
    """
    Renders a standard 2D tactical football pitch canvas (FIFA 105m x 68m standard ratio).
    """
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0, 
                 scale_px_per_m: int = 8, margin_px: int = 40):
        self.length_m = pitch_length_m
        self.width_m = pitch_width_m
        self.scale = scale_px_per_m
        self.margin = margin_px
        
        self.canvas_width = int(self.length_m * self.scale + 2 * self.margin)
        self.canvas_height = int(self.width_m * self.scale + 2 * self.margin)

    def meter_to_canvas(self, point_meters: np.ndarray) -> Tuple[int, int]:
        """
        Converts meter coordinates (X, Y) [0..105, 0..68] to canvas pixel coordinates.
        """
        pts = np.atleast_2d(point_meters)
        x_m = pts[:, 0]
        y_m = pts[:, 1]
        
        canvas_x = (x_m * self.scale + self.margin).astype(int)
        canvas_y = (y_m * self.scale + self.margin).astype(int)
        
        if point_meters.ndim == 1:
            return (int(canvas_x[0]), int(canvas_y[0]))
        return np.column_stack((canvas_x, canvas_y))

    def draw_pitch(self) -> np.ndarray:
        """
        Draws top-down green pitch canvas with standard markings.
        """
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas[:] = (34, 139, 34)  # Forest Green
        
        # Outer boundary rectangle
        pt1 = self.meter_to_canvas(np.array([0, 0]))
        pt2 = self.meter_to_canvas(np.array([105, 68]))
        cv2.rectangle(canvas, pt1, pt2, (255, 255, 255), 2)
        
        # Halfway line
        pt_half_top = self.meter_to_canvas(np.array([52.5, 0]))
        pt_half_bot = self.meter_to_canvas(np.array([52.5, 68]))
        cv2.line(canvas, pt_half_top, pt_half_bot, (255, 255, 255), 2)
        
        # Center circle (9.15m radius)
        pt_center = self.meter_to_canvas(np.array([52.5, 34]))
        circle_radius_px = int(9.15 * self.scale)
        cv2.circle(canvas, pt_center, circle_radius_px, (255, 255, 255), 2)
        cv2.circle(canvas, pt_center, 4, (255, 255, 255), -1)
        
        # Left Penalty Area (16.5m depth, 40.32m width)
        pen_l_top = self.meter_to_canvas(np.array([0, 13.84]))
        pen_l_bot = self.meter_to_canvas(np.array([16.5, 54.16]))
        cv2.rectangle(canvas, pen_l_top, pen_l_bot, (255, 255, 255), 2)
        
        # Right Penalty Area
        pen_r_top = self.meter_to_canvas(np.array([88.5, 13.84]))
        pen_r_bot = self.meter_to_canvas(np.array([105, 54.16]))
        cv2.rectangle(canvas, pen_r_top, pen_r_bot, (255, 255, 255), 2)
        
        return canvas
