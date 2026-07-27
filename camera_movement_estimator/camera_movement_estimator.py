import cv2
import numpy as np
import os
import pickle

class CameraMovementEstimator:
    """
    Camera Movement Estimator matching Abdullah Tarek's football_analysis repository.
    Uses Pyramidal Lucas-Kanade Optical Flow on pitch grass background pixels to estimate frame-by-frame
    camera movement (dx, dy) and adjusts object tracking positions.
    """
    def __init__(self, frame: np.ndarray):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Features to track on background pitch grass
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # Select pitch region (avoiding scoreboards and top broadcast bars)
        mask[int(h * 0.20):int(h * 0.95), :] = 255
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.3, minDistance=30, blockSize=7, mask=mask
        )
        self.features = features

    def add_adjust_positions_to_tracks(self, tracks: dict, camera_movement_per_frame: list):
        """
        Adjusts track positions for players, referees, and ball by subtracting camera displacement.
        """
        from utils.bbox_utils import get_center_of_bbox, get_foot_position
        for object_type, object_tracks in tracks.items():
            for frame_num, track_dict in enumerate(object_tracks):
                for track_id, track_info in track_dict.items():
                    position = track_info.get('position')
                    if position is None:
                        bbox = track_info.get('bbox')
                        if bbox and len(bbox) == 4:
                            position = get_center_of_bbox(bbox) if object_type == "ball" else get_foot_position(bbox)
                        else:
                            continue
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )
                    tracks[object_type][frame_num][track_id]['position'] = position
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames: list, read_from_stub: bool = False, stub_path: str = None):
        """
        Calculates frame-by-frame camera movement (dx, dy). Supports pickle stub caching.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                print(f"[INFO] Loaded camera movement from stub: {stub_path}")
                return pickle.load(f)

        camera_movement = [[0.0, 0.0] for _ in range(len(frames))]
        if len(frames) == 0:
            return camera_movement

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        from tqdm import tqdm
        for frame_num in tqdm(range(1, len(frames)), desc="[5/5] Lucas-Kanade Camera Motion Estimator"):
            frame = frames[frame_num]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.features is None or len(self.features) == 0:
                old_gray = frame_gray.copy()
                continue

            new_features, status, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, self.features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0.0, 0.0

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = self.features[status == 1]

                for new, old in zip(good_new, good_old):
                    diff = new - old
                    dist = np.linalg.norm(diff)
                    if dist > max_distance:
                        max_distance = dist
                        camera_movement_x = diff[0]
                        camera_movement_y = diff[1]

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Update features
                h, w = frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[int(h * 0.20):int(h * 0.95), :] = 255
                self.features = cv2.goodFeaturesToTrack(
                    frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=30, blockSize=7, mask=mask
                )

            old_gray = frame_gray.copy()

        if stub_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
                print(f"[INFO] Saved camera movement stub to: {stub_path}")

        return camera_movement

    def draw_camera_movement(self, frames: list, camera_movement_per_frame: list):
        """
        Draws camera movement text banner on top-left of output frames.
        """
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame_copy = frame.copy()
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (10, 10), (320, 60), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            movement = camera_movement_per_frame[frame_num]
            text_x = f"Camera Movement X: {movement[0]:.2f}"
            text_y = f"Camera Movement Y: {movement[1]:.2f}"
            cv2.putText(frame_copy, text_x, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_copy, text_y, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            output_frames.append(frame_copy)

        return output_frames
