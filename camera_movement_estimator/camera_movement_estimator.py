import os
import pickle
import cv2
import numpy as np

class CameraMovementEstimator:
    """
    Camera Movement Estimator matching Abdullah Tarek's football_analysis repository.
    Uses Pyramidal Lucas-Kanade Optical Flow on pitch grass background pixels to estimate frame-by-frame
    camera movement (dx, dy) and adjusts object tracking positions.
    Supports low-RAM streaming (<500 MB) for long 5-minute+ broadcast clips.
    """
    def __init__(self, frame: np.ndarray = None):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.features = None
        if frame is not None:
            h, w, _ = frame.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(h * 0.20):int(h * 0.95), :] = 255
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.features = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.3, minDistance=30, blockSize=7, mask=mask
            )

    def add_adjust_positions_to_tracks(self, tracks: dict, camera_movement_per_frame: list):
        """
        Adjusts player, referee, and ball coordinates based on accumulated camera movement.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track_dict in enumerate(object_tracks):
                for track_id, track_info in track_dict.items():
                    position = track_info['position']
                    if frame_num >= len(camera_movement_per_frame):
                        if 'position_adjusted' not in tracks[object_type][frame_num][track_id]:
                            tracks[object_type][frame_num][track_id]['position_adjusted'] = position
                        continue
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )
                    tracks[object_type][frame_num][track_id]['position'] = position
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, video_input: str, read_from_stub: bool = False, stub_path: str = None, max_frames: int = None):
        """
        Calculates frame-by-frame camera movement (dx, dy) by streaming video. Supports pickle stub caching.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                print(f"[INFO] Loaded camera movement from stub: {stub_path}")
                return pickle.load(f)

        cap = cv2.VideoCapture(video_input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        camera_movement = [[0.0, 0.0] for _ in range(total_frames)]

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return camera_movement

        # Initialize features on first frame
        h, w, _ = first_frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.20):int(h * 0.95), :] = 255
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.features = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=100, qualityLevel=0.3, minDistance=30, blockSize=7, mask=mask
        )

        from tqdm import tqdm
        pbar = tqdm(total=total_frames - 1, desc="[2/4] Lucas-Kanade Camera Motion Estimator")

        frame_num = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_num >= max_frames):
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.features is None or len(self.features) == 0:
                old_gray = frame_gray.copy()
                pbar.update(1)
                frame_num += 1
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
                if frame_num < len(camera_movement):
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[int(h * 0.20):int(h * 0.95), :] = 255
                self.features = cv2.goodFeaturesToTrack(
                    frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=30, blockSize=7, mask=mask
                )

            old_gray = frame_gray.copy()
            pbar.update(1)
            frame_num += 1

        cap.release()
        pbar.close()

        if stub_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
                print(f"[INFO] Saved camera movement stub to: {stub_path}")

        return camera_movement
