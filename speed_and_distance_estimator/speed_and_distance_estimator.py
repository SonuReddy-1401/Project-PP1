import cv2
import numpy as np
from utils.bbox_utils import measure_distance

class SpeedAndDistanceEstimator:
    """
    Speed and Distance Estimator matching Abdullah Tarek's football_analysis repository.
    Calculates Euclidean distance in pitch meters across a 5-frame window and computes speed in km/h.
    """
    def __init__(self, fps: float = 25.0, frame_window: int = 5):
        self.frame_window = frame_window
        self.fps = fps

    def add_speed_and_distance_to_tracks(self, tracks: dict):
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            if object_type == "ball" or object_type == "referees":
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, track_info in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.fps
                    if time_elapsed <= 0:
                        continue

                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    if object_type not in total_distance:
                        total_distance[object_type] = {}
                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0.0

                    total_distance[object_type][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame + 1):
                        if track_id in object_tracks[frame_num_batch]:
                            object_tracks[frame_num_batch][track_id]['speed'] = speed_km_per_hour
                            object_tracks[frame_num_batch][track_id]['distance'] = total_distance[object_type][track_id]

    def draw_speed_and_distance(self, frames: list, tracks: dict):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type == "ball" or object_type == "referees":
                    continue

                for track_id, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', 0.0)
                        distance = track_info.get('distance', 0.0)

                        bbox = track_info['bbox']
                        position = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))

                        text = f"{speed:.1f} km/h | {distance:.1f}m"
                        cv2.putText(frame, text, (position[0] - 40, position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                        cv2.putText(frame, text, (position[0] - 40, position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            output_frames.append(frame)

        return output_frames
