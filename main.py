import argparse
import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from src.utils.video_utils import get_video_properties, save_video_frames, VideoWriterStream
from src.perception.detector import FootballDetector
from src.perception.tracker import FootballTracker
from src.perception.team_assigner import TeamAssigner
from src.geometry.homography import HomographyTransformer
from src.geometry.pitch_template import TacticalPitchTemplate
from src.analytics.metrics import KinematicMetricsCalculator
from src.analytics.heatmap import PlayerHeatmapGenerator
from src.visualization.drawers import PitchDrawer

def run_pipeline(input_video: str, output_video: str, output_csv: str = None, 
                 config_path: str = "configs/pitch_config.json", max_frames: int = None,
                 model_path: str = "yolov8m.pt", imgsz: int = 1280, conf_threshold: float = 0.15,
                 use_slicing: bool = True, generate_heatmaps: bool = True):
    print("=" * 70)
    print("      INTELLIGENT FOOTBALL PERFORMANCE ANALYSIS PIPELINE      ")
    print("=" * 70)
    
    # 1. Video Properties
    props = get_video_properties(input_video)
    print(f"[INFO] Processing Video: {input_video}")
    print(f"       Resolution: {props['width']}x{props['height']} | FPS: {props['fps']} | Total Frames: {props['total_frames']}")

    # 2. Config & Homography Setup
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg_src = np.array(cfg["default_homography_keypoints"]["src_pixels"], dtype=np.float32)
        dst_m = np.array(cfg["default_homography_keypoints"]["dst_meters"], dtype=np.float32)
        
        # Dynamically scale src_pixels if video resolution differs from reference config resolution (1280x720)
        ref_w = cfg.get("reference_resolution", {}).get("width", 1280)
        ref_h = cfg.get("reference_resolution", {}).get("height", 720)
        scale_x = props['width'] / float(ref_w)
        scale_y = props['height'] / float(ref_h)
        
        src_px = cfg_src.copy()
        src_px[:, 0] *= scale_x
        src_px[:, 1] *= scale_y
    else:
        # Default bounding trapezoid proportional to resolution
        w, h = props['width'], props['height']
        src_px = np.array([[w * 0.08, h * 0.28], [w * 0.92, h * 0.28], [w * 0.98, h * 0.97], [w * 0.02, h * 0.97]], dtype=np.float32)
        dst_m = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]], dtype=np.float32)

    homography = HomographyTransformer(src_px, dst_m)
    
    # 3. Instantiate Perception, Analytics & Visualization Modules
    detector = FootballDetector(model_path=model_path, conf_threshold=conf_threshold, imgsz=imgsz, use_slicing=use_slicing)
    tracker = FootballTracker(track_activation_threshold=conf_threshold)
    team_assigner = TeamAssigner()
    metrics_calc = KinematicMetricsCalculator(fps=props['fps'])
    pitch_drawer = PitchDrawer()
    heatmap_gen = PlayerHeatmapGenerator()

    print(f"       Detection Model: {model_path} | Device: {detector.device.upper()} | Inference Resolution: {imgsz}px | Sliced Tiling: {use_slicing}")

    # 4. Process Video Frame by Frame
    cap = cv2.VideoCapture(input_video)
    video_writer = None
    frame_idx = 0

    total = min(props['total_frames'], max_frames) if max_frames else props['total_frames']
    pbar = tqdm(total=total, desc="[PROCESSING FRAMES]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object Detection & Tracking
        detections = detector.detect_frame(frame)
        players_refs_dets, ball_dets = detector.separate_detections(detections)
        tracked_dets = tracker.update(players_refs_dets)
        
        # Foot positions & Homography Transformation (Pixels -> Meters)
        foot_positions_px = tracker.get_foot_positions(tracked_dets)
        pitch_positions_m = {}
        valid_pitch_positions_m = {}
        for track_id, px in foot_positions_px.items():
            pos_m = homography.pixel_to_pitch(px)[0]
            pitch_positions_m[track_id] = pos_m
            # Check pitch bounds (filters out sideline coaches, sub bench, cameras)
            if homography.is_inside_pitch(pos_m, margin=2.0):
                valid_pitch_positions_m[track_id] = pos_m

        # Continuously update team assigner with CIELAB+HSV crop features and pitch coordinates
        if len(tracked_dets) > 0:
            team_assigner.update(frame, tracked_dets, pitch_positions_m)

        # Update Kinematic Metrics (only for active on-pitch players)
        metrics_calc.update_positions(frame_idx, valid_pitch_positions_m)

        # Draw Annotations
        annotated_frame = pitch_drawer.draw_player_ellipses(frame, tracked_dets, team_assigner, metrics_calc)
        split_view_frame = pitch_drawer.draw_tactical_pitch_overlay(
            annotated_frame, valid_pitch_positions_m, metrics_calc, team_assigner, homography
        )

        # Stream frame directly to disk to maintain O(1) memory usage
        if video_writer is None:
            h_out, w_out, _ = split_view_frame.shape
            video_writer = VideoWriterStream(output_video, fps=props['fps'], frame_size=(w_out, h_out))

        video_writer.write(split_view_frame)
        frame_idx += 1
        pbar.update(1)
        
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    pbar.close()

    # 5. Generate Individual Key Player 2D Pitch Spatial Heatmaps
    if generate_heatmaps:
        out_dir = os.path.dirname(os.path.abspath(output_video))
        heatmap_paths = heatmap_gen.auto_generate_key_player_heatmaps(metrics_calc, team_assigner, output_dir=out_dir)
        print(f"[INFO] Key player 2D pitch spatial heatmaps generated: {heatmap_paths}")

    # 6. Export Metrics Summary CSV (Optional)
    if output_csv:
        df_summary = metrics_calc.export_summary_dataframe()
        df_summary.to_csv(output_csv, index=False)
        print(f"[INFO] Performance metrics summary exported to: {output_csv}")

    print("\n[SUCCESS] Pipeline execution complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Performance Analytics Video Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video clip (.mp4)")
    parser.add_argument("--output", type=str, default="data/output/output_annotated.mp4", help="Path to output video (.mp4)")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to export CSV summary")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Path or name of YOLO model (e.g. yolov8m.pt, yolov8x.pt)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution size (default: 1280 for wide-angle clips)")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames limit")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.csv, max_frames=args.max_frames,
                 model_path=args.model, imgsz=args.imgsz, conf_threshold=args.conf)
