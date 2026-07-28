import os
import argparse
import cv2
import numpy as np

from trackers.tracker import Tracker
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from team_assigner.team_assigner import TeamAssigner
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator

def main(input_video: str = "data/input/match_5min_1080p.mp4",
         output_dir: str = "data/output",
         model_path: str = "models/yolov8x.pt",
         color: str = "red",
         target_team: int = 1,
         max_frames: int = None):

    print("=" * 70)
    print("   ABDULLAH TAREK FOOTBALL ANALYSIS PIPELINE (LOW-RAM STREAMING <500MB)   ")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    out_broadcast = os.path.join(output_dir, "1_broadcast_tracking.mp4")
    out_tactical = os.path.join(output_dir, "2_tactical_pitch_mapping.mp4")

    # 1. Initialize Tracker & Detect Tracks via Stream
    print("[INFO] Running YOLOv8 + ByteTrack Object Tracking (Stream)...")
    tracker = Tracker(model_path=model_path)
    stub_path = os.path.join(output_dir, "stubs_tracks.pkl")
    tracks = tracker.get_object_tracks(input_video, read_from_stub=True, stub_path=stub_path, max_frames=max_frames)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 2. Camera Movement Estimation (Optical Flow Stream)
    print("[INFO] Estimating Camera Motion (Lucas-Kanade Optical Flow Stream)...")
    cam_stub_path = os.path.join(output_dir, "stubs_camera_movement.pkl")
    camera_movement_estimator = CameraMovementEstimator(None)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        input_video, read_from_stub=True, stub_path=cam_stub_path, max_frames=max_frames
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 3. View Transformation (Pixels -> 2D Pitch Canvas Meters)
    print("[INFO] Applying View Transformation (Roboflow Sports Homography Matrix)...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 4. Speed and Distance Estimation
    print("[INFO] Calculating Speed (km/h) & Distance (meters)...")
    speed_and_distance_estimator = SpeedAndDistanceEstimator(fps=25.0)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # 5. Team Assignment (Red Jersey Filtering / KMeans Clustering)
    print(f"[INFO] Assigning Player Teams (Target Color: {color.upper()})...")
    team_assigner = TeamAssigner(target_color=color)
    cap = cv2.VideoCapture(input_video)
    ret, frame_0 = cap.read()
    cap.release()

    if ret and len(tracks["players"]) > 0:
        team_assigner.assign_team_color(frame_0, tracks["players"][0])

    cap = cv2.VideoCapture(input_video)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames) or frame_idx >= len(tracks["players"]):
            break
        player_track = tracks["players"][frame_idx]
        for player_id, track_info in player_track.items():
            team_id = team_assigner.get_player_team(frame, track_info["bbox"], player_id)
            tracks["players"][frame_idx][player_id]["team"] = team_id
        frame_idx += 1
    cap.release()

    # 6. Stream Render Output Videos Directly to File
    print("[INFO] Rendering Output Videos (Direct Stream to MP4)...")
    tracker.render_broadcast_video_stream(input_video, out_broadcast, tracks, camera_movement_per_frame, target_team=target_team)
    tracker.render_tactical_pitch_stream(out_tactical, tracks, target_team=target_team)

    print("\n[SUCCESS] 5-Minute Pipeline execution complete!")
    print(f"Output 1 (Broadcast Video): {out_broadcast}")
    print(f"Output 2 (2D Tactical Pitch Video): {out_tactical}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abdullah Tarek Football Analysis Pipeline")
    parser.add_argument("--input", type=str, default="data/input/new_match_red_team_1080p.mp4", help="Input video path")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Output directory")
    parser.add_argument("--model", type=str, default="models/yolov8x.pt", help="YOLO model weights path")
    parser.add_argument("--color", type=str, default="red", help="Target team jersey color (red)")
    parser.add_argument("--team", type=int, default=1, help="Target team ID (default: 1 for Red Team focus, 0 for all)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames limit")
    args = parser.parse_args()

    main(args.input, args.output_dir, args.model, args.color, args.team, args.max_frames)
