import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv

from src.utils.video_utils import save_video_frames
from src.perception.tracker import FootballTracker
from src.geometry.homography import HomographyTransformer
from src.analytics.metrics import KinematicMetricsCalculator
from src.visualization.drawers import PitchDrawer
from main import run_pipeline

def generate_synthetic_football_video(filename: str = "data/input/synthetic_match.mp4", num_frames: int = 60):
    """
    Generates a synthetic football broadcast clip with grass background and moving pitch markers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    width, height = 1280, 720
    fps = 30.0
    
    frames = []
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # Grass Green
        
        # White pitch border lines
        cv2.rectangle(frame, (80, 100), (1200, 650), (255, 255, 255), 3)
        cv2.line(frame, (640, 100), (640, 650), (255, 255, 255), 3)
        cv2.circle(frame, (640, 375), 70, (255, 255, 255), 3)
        
        frames.append(frame)

    save_video_frames(filename, frames, fps=fps)
    print(f"[TEST SETUP] Generated synthetic football video clip at: {filename}")
    return filename

def test_pipeline_execution():
    print("=" * 70)
    print("      RUNNING END-TO-END PIPELINE & MODULE VERIFICATION       ")
    print("=" * 70)
    
    # 1. Test Module Level Integrations
    print("[TEST 1/2] Verifying Modular Pipeline Geometry & Kinematics...")
    
    src_px = np.array([[100, 200], [1180, 200], [1250, 700], [30, 700]])
    dst_m = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    homography = HomographyTransformer(src_px, dst_m)
    
    # Verify homography transform
    pt_px = np.array([[640.0, 450.0]])
    pt_m = homography.pixel_to_pitch(pt_px)
    assert pt_m.shape == (1, 2), "Homography output shape mismatch!"
    print(f"       Pixel {pt_px[0]} -> Pitch Coordinate: ({pt_m[0][0]:.2f}m, {pt_m[0][1]:.2f}m)")
    
    # Verify metrics calculator
    metrics = KinematicMetricsCalculator(fps=30.0)
    for f_idx in range(30):
        # Simulate player 1 moving forward 0.2 meters per frame
        metrics.update_positions(f_idx, {1: np.array([10.0 + f_idx * 0.2, 20.0])})
        
    stats = metrics.get_player_stats(1)
    print(f"       Simulated Player #1 Stats: {stats}")
    assert stats["total_distance_m"] > 5.0, "Distance metric calculation error!"
    assert stats["current_speed_kmh"] > 0.0, "Speed calculation error!"

    # Verify drawers
    drawer = PitchDrawer()
    pitch_img = drawer.pitch_template.draw_pitch()
    assert pitch_img is not None and pitch_img.shape[0] > 0, "Pitch drawer error!"

    # 2. Test End-to-End Main Video Pipeline Execution
    print("\n[TEST 2/2] Running Full Video Pipeline Integration...")
    input_video = generate_synthetic_football_video()
    output_video = "data/output/synthetic_output.mp4"
    output_csv = "data/output/synthetic_metrics.csv"
    
    run_pipeline(input_video=input_video, output_video=output_video, output_csv=output_csv, max_frames=30, model_path="yolov8n.pt", imgsz=640)
    
    assert os.path.exists(output_video), "Output video file was not generated!"
    assert os.path.exists(output_csv), "Output CSV file was not generated!"
    
    df = pd.read_csv(output_csv)
    print("\n[EXPORT CHECK] Output CSV dataframe structure:")
    print(df)
    print("\n[PASS] All Pipeline Integration & Verification Tests Completed Successfully!")

if __name__ == "__main__":
    test_pipeline_execution()
