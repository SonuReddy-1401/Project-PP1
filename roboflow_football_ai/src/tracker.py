import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

class FootballAITracker:
    """
    Roboflow AI (football-ai.ipynb) Tracker implementation.
    Streamed low-RAM (<500 MB RAM) frame processing with ByteTrack,
    team color assignment, player head badges, and 2D tactical pitch radar rendering.
    """
    def __init__(self, model_path: str = "models/yolov8x.pt"):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.08)

    def detect_and_track(self, video_path: str, max_frames: int = None):
        """
        Executes YOLO object detection and ByteTrack tracking on video frames.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        tracks = {"players": [], "referees": [], "ball": []}
        batch_frames = []
        frame_idx = 0

        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="[1/3] Roboflow AI YOLOv8 + ByteTrack Object Tracking")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            batch_frames.append(frame)
            frame_idx += 1

            if len(batch_frames) == 20:
                results = self.model.predict(batch_frames, conf=0.10, imgsz=1280, verbose=False)

                for res in results:
                    cls_names = res.names
                    cls_names_inv = {v: k for k, v in cls_names.items()}
                    supervision_dets = sv.Detections.from_ultralytics(res)

                    # Treat goalkeeper as person class
                    for idx_obj, cls_id in enumerate(supervision_dets.class_id):
                        if cls_names[cls_id] == "goalkeeper":
                            supervision_dets.class_id[idx_obj] = cls_names_inv.get("person", 0)

                    tracked_dets = self.tracker.update_with_detections(supervision_dets)

                    frame_players = {}
                    frame_referees = {}
                    frame_ball = {}

                    for det in tracked_dets:
                        bbox = det[0].tolist()
                        cls_id = det[3]
                        track_id = det[4]

                        if cls_id == cls_names_inv.get("person", 0):
                            foot_pos = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
                            frame_players[track_id] = {"bbox": bbox, "position": foot_pos, "crop": sv.crop_image(res.orig_img, det[0])}
                        elif cls_id == cls_names_inv.get("referee", 1):
                            foot_pos = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
                            frame_referees[track_id] = {"bbox": bbox, "position": foot_pos}

                    # Extract ball
                    for det in supervision_dets:
                        bbox = det[0].tolist()
                        cls_id = det[3]
                        if cls_id == cls_names_inv.get("sports ball", 32):
                            center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
                            frame_ball[1] = {"bbox": bbox, "position": center}
                            break

                    tracks["players"].append(frame_players)
                    tracks["referees"].append(frame_referees)
                    tracks["ball"].append(frame_ball)

                pbar.update(len(batch_frames))
                batch_frames = []

        if len(batch_frames) > 0:
            results = self.model.predict(batch_frames, conf=0.10, imgsz=1280, verbose=False)
            for res in results:
                cls_names = res.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                supervision_dets = sv.Detections.from_ultralytics(res)

                for idx_obj, cls_id in enumerate(supervision_dets.class_id):
                    if cls_names[cls_id] == "goalkeeper":
                        supervision_dets.class_id[idx_obj] = cls_names_inv.get("person", 0)

                tracked_dets = self.tracker.update_with_detections(supervision_dets)

                frame_players = {}
                frame_referees = {}
                frame_ball = {}

                for det in tracked_dets:
                    bbox = det[0].tolist()
                    cls_id = det[3]
                    track_id = det[4]

                    if cls_id == cls_names_inv.get("person", 0):
                        foot_pos = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
                        frame_players[track_id] = {"bbox": bbox, "position": foot_pos, "crop": sv.crop_image(res.orig_img, det[0])}
                    elif cls_id == cls_names_inv.get("referee", 1):
                        foot_pos = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
                        frame_referees[track_id] = {"bbox": bbox, "position": foot_pos}

                for det in supervision_dets:
                    bbox = det[0].tolist()
                    cls_id = det[3]
                    if cls_id == cls_names_inv.get("sports ball", 32):
                        center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
                        frame_ball[1] = {"bbox": bbox, "position": center}
                        break

                tracks["players"].append(frame_players)
                tracks["referees"].append(frame_referees)
                tracks["ball"].append(frame_ball)

            pbar.update(len(batch_frames))

        cap.release()
        pbar.close()

        # Ball trajectory interpolation (limit 15 frames max)
        ball_positions_list = [x.get(1, {}).get('bbox', [np.nan, np.nan, np.nan, np.nan]) for x in tracks["ball"]]
        df_ball = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball = df_ball.interpolate(method='linear', limit=15, limit_direction='both')

        for idx, row in enumerate(df_ball.to_numpy().tolist()):
            if not any(np.isnan(row)):
                tracks["ball"][idx][1] = {
                    "bbox": row,
                    "position": [(row[0] + row[2]) / 2.0, (row[1] + row[3]) / 2.0]
                }
            else:
                tracks["ball"][idx] = {}

        return tracks

    def render_broadcast_video(self, video_input: str, output_video: str, tracks: dict):
        from tqdm import tqdm
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(len(tracks["players"]), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for frame_num in tqdm(range(total_frames), desc="[2/3] Streaming Output 1: Broadcast Tracking Video"):
            ret, frame = cap.read()
            if not ret:
                break

            players = tracks["players"][frame_num]
            for track_id, player in players.items():
                bbox = player["bbox"]
                team_id = player.get("team", 0)
                color = (0, 0, 255) if team_id == 0 else (255, 255, 255)

                x1, y1, x2, y2 = map(int, bbox)
                x_center = (x1 + x2) // 2

                # Ground ellipse
                w = x2 - x1
                cv2.ellipse(frame, (x_center, y2), (int(w), int(0.35 * w)), 0, -45, 235, color, 2, cv2.LINE_AA)

                # Head badge ABOVE THE HEAD (y1 - 25)
                badge = f"ID:{track_id}"
                (fw, fh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                rx1, ry1 = x_center - fw // 2 - 4, y1 - fh - 12
                rx2, ry2 = x_center + fw // 2 + 4, y1 - 4

                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 0), cv2.FILLED)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1)
                cv2.putText(frame, badge, (rx1 + 4, ry2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Ball pointer
            ball_dict = tracks["ball"][frame_num]
            for _, ball in ball_dict.items():
                bbox = ball["bbox"]
                bx = int((bbox[0] + bbox[2]) / 2.0)
                by = int(bbox[1])
                tri = np.array([[bx, by], [bx - 10, by - 20], [bx + 10, by - 20]])
                cv2.drawContours(frame, [tri], 0, (0, 255, 0), cv2.FILLED)

            out.write(frame)

        cap.release()
        out.release()

    def render_tactical_pitch_video(self, output_video: str, tracks: dict, fps: float = 25.0):
        from tqdm import tqdm
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)

        pitch_w, pitch_h = 1050, 680
        total_frames = len(tracks["players"])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (pitch_w, pitch_h))

        for frame_num in tqdm(range(total_frames), desc="[3/3] Streaming Output 2: 2D Tactical Pitch Video"):
            canvas = np.full((pitch_h, pitch_w, 3), (34, 139, 34), dtype=np.uint8)

            cv2.rectangle(canvas, (50, 40), (1000, 640), (255, 255, 255), 2)
            cv2.line(canvas, (525, 40), (525, 640), (255, 255, 255), 2)
            cv2.circle(canvas, (525, 340), 90, (255, 255, 255), 2)
            cv2.rectangle(canvas, (50, 190), (200, 490), (255, 255, 255), 2)
            cv2.rectangle(canvas, (850, 190), (1000, 490), (255, 255, 255), 2)

            players = tracks["players"][frame_num]
            for track_id, player in players.items():
                pos_m = player.get("position_transformed")
                if not pos_m:
                    continue

                px = int(50 + (pos_m[0] / 105.0) * 950.0)
                py = int(40 + (pos_m[1] / 68.0) * 600.0)

                team_id = player.get("team", 0)
                color = (0, 0, 255) if team_id == 0 else (255, 255, 255)

                cv2.circle(canvas, (px, py), 12, color, -1)
                cv2.circle(canvas, (px, py), 12, (0, 0, 0), 2)

                tx = px - 6 if track_id < 10 else px - 10
                cv2.putText(canvas, f"{track_id}", (tx, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

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
