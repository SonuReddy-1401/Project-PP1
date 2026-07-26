import warnings
import numpy as np
import supervision as sv
from typing import Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

class FootballTracker:
    """
    Multi-object tracking wrapper utilizing Supervision ByteTrack.
    Calibrated for small target persistence across wide-angle broadcast frames.
    Preserves unique player track IDs across frames.
    """
    def __init__(self, track_activation_threshold: float = 0.15, lost_track_buffer: int = 45, minimum_matching_threshold: float = 0.7):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates tracker state with new frame detections.
        Returns supervision Detections object with populated tracker_id.
        """
        if len(detections) == 0:
            return detections
            
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections

    def get_foot_positions(self, tracked_detections: sv.Detections) -> Dict[int, np.ndarray]:
        """
        Extracts bottom-center foot coordinates (x, y) for each tracked entity.
        Returns mapping: track_id -> np.array([x_bottom, y_bottom])
        """
        foot_positions = {}
        if tracked_detections.tracker_id is None:
            return foot_positions

        for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2.0
            y_bottom = y2  # foot position on the grass plane
            foot_positions[int(track_id)] = np.array([x_center, y_bottom])
            
        return foot_positions
