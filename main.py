import os
import argparse
import cv2
import numpy as np

from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from team_assigner.team_assigner import TeamAssigner
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator

def main(input_video: str = "data/input/new_match_red_team_1080p.mp4",
         output_dir: str = "data/output",
         model_path: str = "models/yolov8x.pt",
         color: str = "red",
         target_team: int = 1,
         max_frames: int = None):

    print("=" * 70)
    print("   ABDULLAH TAREK FOOTBALL ANALYSIS PIPELINE RECREATION   ")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    out_broadcast = os.path.join(output_dir, "1_broadcast_tracking.mp4")
    out_tactical = os.path.join(output_dir, "2_tactical_pitch_mapping.mp4")

    # 1. Read Video Frames
    print(f"[INFO] Reading video frames: {input_video}")
    video_frames = read_video(input_video)
    if max_frames and len(video_frames) > max_frames:
        video_frames = video_frames[:max_frames]
        print(f"[INFO] Limiting processing to {max_frames} frames")

    if not video_frames:
        print("[ERROR] No frames loaded!")
        return

    # 2. Initialize Tracker & Detect Tracks
    print("[INFO] Running YOLOv8 + ByteTrack Object Tracking...")
    tracker = Tracker(model_path=model_path)
    stub_path = os.path.join(output_dir, "stubs_tracks.pkl")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 3. Camera Movement Estimation (Optical Flow)
    print("[INFO] Estimating Camera Motion (Lucas-Kanade Optical Flow)...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    cam_stub_path = os.path.join(output_dir, "stubs_camera_movement.pkl")
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path=cam_stub_path
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 4. View Transformation (Pixels -> 2D Pitch Canvas Meters)
    print("[INFO] Applying View Transformation (Homography Perspective Matrix)...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 5. Speed and Distance Estimation
    print("[INFO] Calculating Speed (km/h) & Distance (meters)...")
    speed_and_distance_estimator = SpeedAndDistanceEstimator(fps=25.0)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # 6. Team Assignment (Red Jersey Filtering / KMeans Clustering)
    print(f"[INFO] Assigning Player Teams (Target Color: {color.upper()})...")
    team_assigner = TeamAssigner(target_color=color)
    if len(tracks["players"]) > 0:
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track_info in player_track.items():
            team_id = team_assigner.get_player_team(
                video_frames[frame_num], track_info["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team_id

    # 7. Draw Annotations & Save Output Videos
    print("[INFO] Drawing Annotations & Rendering Output Videos...")

    # Output 1: Broadcast Video
    output_broadcast_frames = tracker.draw_annotations(video_frames, tracks, target_team=target_team)
    output_broadcast_frames = camera_movement_estimator.draw_camera_movement(output_broadcast_frames, camera_movement_per_frame)
    output_broadcast_frames = speed_and_distance_estimator.draw_speed_and_distance(output_broadcast_frames, tracks)
    save_video(output_broadcast_frames, out_broadcast)

    # Output 2: Standalone 2D Tactical Pitch Video
    output_tactical_frames = tracker.draw_2d_tactical_pitch(video_frames, tracks, target_team=target_team)
    save_video(output_tactical_frames, out_tactical)

    print("\n[SUCCESS] Pipeline execution complete!")
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
