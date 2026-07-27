import cv2
import numpy as np

class ViewTransformer:
    """
    View Transformer matching Abdullah Tarek's football_analysis repository.
    Applies 4-point perspective transformation to convert screen foot coordinates (pixels)
    to top-down 2D tactical pitch canvas coordinates (meters).
    """
    def __init__(self, pixel_vertices=None, target_vertices=None):
        if pixel_vertices is None:
            # Default trapezoid source polygon in 1080p screen space
            pixel_vertices = np.array([
                [0.0, 318.0],
                [1919.0, 318.0],
                [1919.0, 1079.0],
                [0.0, 1079.0]
            ], dtype=np.float32)

        if target_vertices is None:
            # FIFA Standard 105m x 68m Pitch Canvas Coordinates (mapped to 0-105m x 0-68m)
            target_vertices = np.array([
                [15.0, 0.0],
                [90.0, 0.0],
                [90.0, 68.0],
                [15.0, 68.0]
            ], dtype=np.float32)

        self.pixel_vertices = pixel_vertices.astype(np.float32)
        self.target_vertices = target_vertices.astype(np.float32)

        self.perpective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        p = int(point[0]), int(point[1])
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perpective_transform)
        return transformed_point.reshape(-1, 2)[0]

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
                    if position_transformed is not None:
                        position_transformed = position_transformed.tolist()
                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
