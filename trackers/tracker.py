import os
import pickle
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from utils.bbox_utils import get_center_of_bbox, get_foot_position, get_bbox_width

class Tracker:
    """
    Tracker matching Abdullah Tarek's football_analysis repository.
    Uses YOLOv8 + ByteTrack to track players, referees, and the ball.
    Draws player ground ellipses, team color markers, track IDs, ball triangle pointers, and 2D pitch map overlays.
    """
    def __init__(self, model_path: str = "models/yolov8x.pt"):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.10)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions_list = [x.get(1, {}).get('bbox', [np.nan, np.nan, np.nan, np.nan]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        new_ball_positions = []
        for row in df_ball_positions.to_numpy().tolist():
            bbox = row
            center = get_center_of_bbox(bbox) if not any(np.isnan(bbox)) else [0, 0]
            new_ball_positions.append({1: {"bbox": bbox, "position": center}})
        return new_ball_positions

    def detect_frames(self, frames):
        from tqdm import tqdm
        batch_size = 20
        detections = []
        for i in tqdm(range(0, len(frames), batch_size), desc="[1/5] YOLOv8 GPU Object Detection"):
            batch_frames = frames[i:i + batch_size]
            results = self.model.predict(batch_frames, conf=0.10, imgsz=1280, verbose=False)
            detections.extend(results)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                print(f"[INFO] Loaded tracks from stub: {stub_path}")
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        from tqdm import tqdm
        for frame_num, detection in enumerate(tqdm(detections, desc="[2/5] ByteTrack Player & Ball Tracking")):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to Supervision Detection
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to Player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv.get("person", 0)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get("person", 0):
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": get_foot_position(bbox)
                    }

                if cls_id == cls_names_inv.get("referee", 1):
                    tracks["referees"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": get_foot_position(bbox)
                    }

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv.get("sports ball", 32):
                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox,
                        "position": get_center_of_bbox(bbox)
                    }

        if stub_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                print(f"[INFO] Saved tracks stub to: {stub_path}")

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            x2_rect = x_center + rectangle_width // 2
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            text_x = x1_rect + 12 if track_id < 10 else x1_rect + 6
            cv2.putText(
                frame,
                f"{track_id}",
                (text_x, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame.count(1)
        team_2_num_frames = team_ball_control_till_frame.count(2)

        total_frames = team_1_num_frames + team_2_num_frames + 1e-5
        team_1_ratio = team_1_num_frames / total_frames
        team_2_ratio = team_2_num_frames / total_frames

        cv2.putText(frame, f"Team 1 (Red) Possession: {team_1_ratio * 100:.1f}%", (1370, 890), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Team 2 (White) Possession: {team_2_ratio * 100:.1f}%", (1370, 940), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None, target_team=1):
        from tqdm import tqdm
        output_video_frames = []

        for frame_num, frame in enumerate(tqdm(video_frames, desc="[3/5] Rendering Broadcast Output Frames")):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                team_id = player.get("team", 1)

                # Filter target team if requested
                if target_team is not None and target_team != 0 and team_id != target_team:
                    continue

                color = (0, 0, 255) if team_id == 1 else (255, 255, 255)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw Possession stats
            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    def draw_2d_tactical_pitch(self, video_frames, tracks, target_team=1):
        """
        Renders a top-down standalone 2D tactical pitch video showing player dots.
        """
        from tqdm import tqdm
        output_tactical_frames = []
        pitch_w, pitch_h = 1050, 680

        for frame_num, frame in enumerate(tqdm(video_frames, desc="[4/5] Rendering 2D Tactical Pitch Canvas")):
            # Create green pitch canvas
            canvas = np.full((pitch_h, pitch_w, 3), (34, 139, 34), dtype=np.uint8)
            
            # Outer touchlines
            cv2.rectangle(canvas, (50, 40), (1000, 640), (255, 255, 255), 2)
            # Center line & circle
            cv2.line(canvas, (525, 40), (525, 640), (255, 255, 255), 2)
            cv2.circle(canvas, (525, 340), 90, (255, 255, 255), 2)
            # Penalty boxes
            cv2.rectangle(canvas, (50, 190), (200, 490), (255, 255, 255), 2)
            cv2.rectangle(canvas, (850, 190), (1000, 490), (255, 255, 255), 2)

            player_dict = tracks["players"][frame_num]
            for track_id, player in player_dict.items():
                team_id = player.get("team", 1)
                if target_team is not None and target_team != 0 and team_id != target_team:
                    continue

                pos_m = player.get("position_transformed")
                if pos_m is None:
                    continue

                # Map 105m x 68m meters to canvas pixels (50..1000 x, 40..640 y)
                px_x = int(50 + (pos_m[0] / 105.0) * 950.0)
                px_y = int(40 + (pos_m[1] / 68.0) * 600.0)

                color = (0, 0, 255) if team_id == 1 else (255, 255, 255)
                cv2.circle(canvas, (px_x, px_y), 12, color, -1)
                cv2.circle(canvas, (px_x, px_y), 12, (0, 0, 0), 2)

                text_x = px_x - 6 if track_id < 10 else px_x - 10
                cv2.putText(canvas, f"{track_id}", (text_x, px_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            output_tactical_frames.append(canvas)

        return output_tactical_frames
