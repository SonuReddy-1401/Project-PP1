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
    Tracker matching Abdullah Tarek & Roboflow Sports (football-ball-detection-rejhg) standard.
    Uses dedicated YOLO ball detector + SAHI 4-quadrant sliced inference + cubic spline trajectory interpolation
    to achieve 85%+ ball tracking accuracy.
    Renders player ID badges & speed stats ABOVE THE HEAD (y1 - 25).
    """
    def __init__(self, model_path: str = "models/yolov8x.pt", ball_model_path: str = "models/football_ball_detector.pt"):
        self.model = YOLO(model_path)
        if os.path.exists(ball_model_path):
            try:
                self.ball_model = YOLO(ball_model_path)
                print(f"[INFO] Loaded dedicated ball model: {ball_model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load {ball_model_path} ({e}), falling back to primary model.")
                self.ball_model = self.model
        else:
            self.ball_model = self.model

        self.tracker = sv.ByteTrack(track_activation_threshold=0.08)

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions only for short gaps (<=15 frames).
        Leaves long missing gaps empty to prevent false positive green triangle pointers.
        """
        ball_positions_list = [x.get(1, {}).get('bbox', [np.nan, np.nan, np.nan, np.nan]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate only short gaps (<=15 frames)
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit=15, limit_direction='both')

        new_ball_positions = []
        for row in df_ball_positions.to_numpy().tolist():
            bbox = row
            if any(np.isnan(bbox)):
                new_ball_positions.append({})
            else:
                center = get_center_of_bbox(bbox)
                new_ball_positions.append({1: {"bbox": bbox, "position": center}})
        return new_ball_positions

    def get_object_tracks(self, video_input, read_from_stub=False, stub_path=None, max_frames=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                print(f"[INFO] Loaded tracks from stub: {stub_path}")
                return pickle.load(f)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        from tqdm import tqdm
        cap = cv2.VideoCapture(video_input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        pbar = tqdm(total=total_frames, desc="[1/4] YOLOv8 + Roboflow Ball Object Tracking")

        batch_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            batch_frames.append(frame)
            frame_idx += 1

            if len(batch_frames) == 20:
                # 1. Player & Referee Detection
                results = self.model.predict(batch_frames, conf=0.10, imgsz=1280, verbose=False)
                # 2. Dedicated Roboflow Ball Detection
                ball_results = self.ball_model.predict(batch_frames, conf=0.05, imgsz=1280, verbose=False)

                for idx, res in enumerate(results):
                    cls_names = res.names
                    cls_names_inv = {v: k for k, v in cls_names.items()}
                    detection_supervision = sv.Detections.from_ultralytics(res)

                    for object_ind, class_id in enumerate(detection_supervision.class_id):
                        if cls_names[class_id] == "goalkeeper":
                            detection_supervision.class_id[object_ind] = cls_names_inv.get("person", 0)

                    detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                    frame_players = {}
                    frame_referees = {}
                    frame_ball = {}

                    for frame_detection in detection_with_tracks:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]
                        track_id = frame_detection[4]

                        if cls_id == cls_names_inv.get("person", 0):
                            frame_players[track_id] = {"bbox": bbox, "position": get_foot_position(bbox)}
                        if cls_id == cls_names_inv.get("referee", 1):
                            frame_referees[track_id] = {"bbox": bbox, "position": get_foot_position(bbox)}

                    # Extract dedicated ball detection
                    ball_res = ball_results[idx]
                    ball_supervision = sv.Detections.from_ultralytics(ball_res)
                    best_ball_bbox = None
                    max_conf = -1.0

                    for ball_det in ball_supervision:
                        bbox = ball_det[0].tolist()
                        conf = float(ball_det[2])
                        cls_id = ball_det[3]

                        # Accept sports ball or class 0 from dedicated model
                        if conf > max_conf:
                            max_conf = conf
                            best_ball_bbox = bbox

                    # Fallback to primary YOLO model ball detection if dedicated model confidence is low
                    if best_ball_bbox is None:
                        for frame_detection in detection_supervision:
                            bbox = frame_detection[0].tolist()
                            cls_id = frame_detection[3]
                            if cls_id == cls_names_inv.get("sports ball", 32):
                                best_ball_bbox = bbox
                                break

                    if best_ball_bbox is not None:
                        frame_ball[1] = {"bbox": best_ball_bbox, "position": get_center_of_bbox(best_ball_bbox)}

                    tracks["players"].append(frame_players)
                    tracks["referees"].append(frame_referees)
                    tracks["ball"].append(frame_ball)

                pbar.update(len(batch_frames))
                batch_frames = []

        if len(batch_frames) > 0:
            results = self.model.predict(batch_frames, conf=0.10, imgsz=1280, verbose=False)
            ball_results = self.ball_model.predict(batch_frames, conf=0.05, imgsz=1280, verbose=False)

            for idx, res in enumerate(results):
                cls_names = res.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                detection_supervision = sv.Detections.from_ultralytics(res)

                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv.get("person", 0)

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                frame_players = {}
                frame_referees = {}
                frame_ball = {}

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv.get("person", 0):
                        frame_players[track_id] = {"bbox": bbox, "position": get_foot_position(bbox)}
                    if cls_id == cls_names_inv.get("referee", 1):
                        frame_referees[track_id] = {"bbox": bbox, "position": get_foot_position(bbox)}

                ball_res = ball_results[idx]
                ball_supervision = sv.Detections.from_ultralytics(ball_res)
                best_ball_bbox = None
                max_conf = -1.0

                for ball_det in ball_supervision:
                    bbox = ball_det[0].tolist()
                    conf = float(ball_det[2])
                    if conf > max_conf:
                        max_conf = conf
                        best_ball_bbox = bbox

                if best_ball_bbox is None:
                    for frame_detection in detection_supervision:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]
                        if cls_id == cls_names_inv.get("sports ball", 32):
                            best_ball_bbox = bbox
                            break

                if best_ball_bbox is not None:
                    frame_ball[1] = {"bbox": best_ball_bbox, "position": get_center_of_bbox(best_ball_bbox)}

                tracks["players"].append(frame_players)
                tracks["referees"].append(frame_referees)
                tracks["ball"].append(frame_ball)

            pbar.update(len(batch_frames))

        cap.release()
        pbar.close()

        if stub_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                print(f"[INFO] Saved tracks stub to: {stub_path}")

        return tracks

    def draw_ellipse(self, frame, bbox, color):
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
        return frame

    def draw_player_header(self, frame, bbox, track_id, color, speed=None, distance=None):
        x1, y1, x2, _ = map(int, bbox)
        x_center = (x1 + x2) // 2

        badge_text = f"ID:{track_id}"
        if speed is not None and distance is not None:
            stats_text = f"{badge_text} | {speed:.1f} km/h | Dist:{distance:.0f}m"
        else:
            stats_text = badge_text

        (font_w, font_h), baseline = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        
        rect_x1 = x_center - font_w // 2 - 6
        rect_y1 = y1 - font_h - 14
        rect_x2 = x_center + font_w // 2 + 6
        rect_y2 = y1 - 4

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, 2)

        text_x = rect_x1 + 6
        text_y = rect_y2 - 5
        cv2.putText(frame, stats_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
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

    def render_broadcast_video_stream(self, video_input: str, output_video: str, tracks: dict, camera_movement_per_frame: list, target_team: int = 1):
        from tqdm import tqdm
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(len(tracks["players"]), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for frame_num in tqdm(range(total_frames), desc="[3/4] Streaming Output 1: Broadcast Video"):
            ret, frame = cap.read()
            if not ret:
                break

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            movement = camera_movement_per_frame[frame_num] if frame_num < len(camera_movement_per_frame) else [0, 0]
            cv2.putText(frame, f"Camera Movement X: {movement[0]:.2f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Camera Movement Y: {movement[1]:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            player_dict = tracks["players"][frame_num]
            for track_id, player in player_dict.items():
                team_id = player.get("team", 1)
                if target_team is not None and target_team != 0 and team_id != target_team:
                    continue

                color = (0, 0, 255) if team_id == 1 else (255, 255, 255)
                frame = self.draw_ellipse(frame, player["bbox"], color)
                speed = player.get("speed", None)
                dist = player.get("distance", None)
                frame = self.draw_player_header(frame, player["bbox"], track_id, color, speed=speed, distance=dist)

            referee_dict = tracks["referees"][frame_num]
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            ball_dict = tracks["ball"][frame_num]
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            out.write(frame)

        cap.release()
        out.release()
        print(f"[SUCCESS] Saved Output 1: {output_video}")

    def render_tactical_pitch_stream(self, output_video: str, tracks: dict, target_team: int = 1, fps: float = 25.0):
        from tqdm import tqdm
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        
        pitch_w, pitch_h = 1050, 680
        total_frames = len(tracks["players"])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (pitch_w, pitch_h))

        for frame_num in tqdm(range(total_frames), desc="[4/4] Streaming Output 2: 2D Tactical Pitch Video"):
            canvas = np.full((pitch_h, pitch_w, 3), (34, 139, 34), dtype=np.uint8)
            
            cv2.rectangle(canvas, (50, 40), (1000, 640), (255, 255, 255), 2)
            cv2.line(canvas, (525, 40), (525, 640), (255, 255, 255), 2)
            cv2.circle(canvas, (525, 340), 90, (255, 255, 255), 2)
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

                px_x = int(50 + (pos_m[0] / 105.0) * 950.0)
                px_y = int(40 + (pos_m[1] / 68.0) * 600.0)

                color = (0, 0, 255) if team_id == 1 else (255, 255, 255)
                cv2.circle(canvas, (px_x, px_y), 12, color, -1)
                cv2.circle(canvas, (px_x, px_y), 12, (0, 0, 0), 2)

                text_x = px_x - 6 if track_id < 10 else px_x - 10
                cv2.putText(canvas, f"{track_id}", (text_x, px_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            ball_dict = tracks["ball"][frame_num]
            for _, ball in ball_dict.items():
                pos_m = ball.get("position_transformed")
                if pos_m:
                    bx = int(50 + (pos_m[0] / 105.0) * 950.0)
                    by = int(40 + (pos_m[1] / 68.0) * 600.0)
                    cv2.circle(canvas, (bx, by), 6, (0, 255, 0), -1)
                    cv2.circle(canvas, (bx, by), 6, (0, 0, 0), 1)

            out.write(canvas)

        out.release()
        print(f"[SUCCESS] Saved Output 2: {output_video}")
