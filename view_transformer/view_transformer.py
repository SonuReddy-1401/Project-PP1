import cv2
import numpy as np

class SoccerPitchConfiguration:
    """
    Standard FIFA Soccer Pitch Configuration matching Roboflow Sports (roboflow/sports).
    Defines physical dimensions (105m x 68m) and keypoint landmarks in meters.
    """
    def __init__(self, length_meters: float = 105.0, width_meters: float = 68.0):
        self.length = length_meters
        self.width = width_meters

class ViewTransformer:
    """
    Roboflow Sports Compatible Perspective View Transformer.
    Applies 4-point homography perspective transformation matrix H to convert screen foot coordinates (pixels)
    to top-down 2D tactical pitch canvas coordinates (meters).
    """
    def __init__(self, pixel_vertices=None, target_vertices=None):
        self.config = SoccerPitchConfiguration()

        if pixel_vertices is None:
            # Broadcast camera source polygon in 1080p screen space
            pixel_vertices = np.array([
                [110.0, 340.0],
                [1810.0, 340.0],
                [1919.0, 1079.0],
                [0.0, 1079.0]
            ], dtype=np.float32)

        if target_vertices is None:
            # FIFA Standard Pitch Metric Canvas Coordinates (0..105m x, 0..68m y)
            target_vertices = np.array([
                [0.0, 0.0],
                [105.0, 0.0],
                [105.0, 68.0],
                [0.0, 68.0]
            ], dtype=np.float32)

        self.pixel_vertices = pixel_vertices.astype(np.float32)
        self.target_vertices = target_vertices.astype(np.float32)

        self.perpective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """
        Transforms screen pixel coordinate (x, y) into 2D tactical pitch canvas meters (0..105m, 0..68m).
        Clamps values smoothly within pitch bounds.
        """
        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perpective_transform)
        pt = transformed_point.reshape(-1, 2)[0]
        
        # Clamp to FIFA pitch canvas bounds
        px_x = float(np.clip(pt[0], 0.0, 105.0))
        px_y = float(np.clip(pt[1], 0.0, 68.0))
        return [px_x, px_y]

    def add_transformed_position_to_tracks(self, tracks: dict):
        """
        Adds 2D tactical pitch position (in pitch meters) to player, referee, and ball tracks.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track_dict in enumerate(object_tracks):
                for track_id, track_info in track_dict.items():
                    position = track_info.get('position_adjusted', track_info['position'])
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
