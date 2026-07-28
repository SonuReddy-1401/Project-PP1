import argparse
import os
import sys
import numpy as np

# Ensure local package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tracker import FootballAITracker
from src.team_classifier import TeamClassifier
from src.pitch_transformer import PitchTransformer

def main():
    parser = argparse.ArgumentParser(description="Roboflow Football AI (football-ai.ipynb) Pipeline")
    parser.add_argument("--input", type=str, default="roboflow_football_ai/input/new_match_red_team_1080p.mp4")
    parser.add_argument("--output_dir", type=str, default="roboflow_football_ai/output")
    parser.add_argument("--model", type=str, default="models/yolov8x.pt")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("      ROBOFLOW FOOTBALL AI (FOOTBALL-AI.IPYNB) ISOLATED PIPELINE      ")
    print("=" * 70)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input video not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Track objects using ByteTrack & YOLOv8
    tracker = FootballAITracker(model_path=args.model)
    tracks = tracker.detect_and_track(args.input, max_frames=args.max_frames)

    # 2. Team Classifier Fit & Predict
    print("[INFO] Fitting Roboflow TeamClassifier on player crops...")
    crops = []
    for frame_players in tracks["players"]:
        for p_id, p_data in frame_players.items():
            if "crop" in p_data:
                crops.append(p_data["crop"])

    team_classifier = TeamClassifier(device="cuda")
    if len(crops) > 0:
        team_classifier.fit(crops[:100])

    for frame_players in tracks["players"]:
        for p_id, p_data in frame_players.items():
            if "crop" in p_data:
                team_id = team_classifier.predict(p_data["crop"])
                p_data["team"] = team_id

    # 3. 2D Pitch View Transformer
    print("[INFO] Applying Roboflow 32-Keypoint Homography ViewTransformer...")
    pitch_transformer = PitchTransformer()

    for frame_players in tracks["players"]:
        for p_id, p_data in frame_players.items():
            p_data["position_transformed"] = pitch_transformer.transform_point(p_data["position"])

    for frame_ball in tracks["ball"]:
        for b_id, b_data in frame_ball.items():
            b_data["position_transformed"] = pitch_transformer.transform_point(b_data["position"])

    # 4. Render Output 1 (Broadcast Video)
    out1_path = os.path.join(args.output_dir, "1_broadcast_tracking.mp4")
    tracker.render_broadcast_video(args.input, out1_path, tracks)

    # 5. Render Output 2 (2D Tactical Pitch Video)
    out2_path = os.path.join(args.output_dir, "2_tactical_pitch_mapping.mp4")
    tracker.render_tactical_pitch_video(out2_path, tracks)

    print("\n" + "=" * 70)
    print("      ROBOFLOW FOOTBALL AI PIPELINE COMPLETE!      ")
    print(f"[SUCCESS] Output 1 (Broadcast): {out1_path}")
    print(f"[SUCCESS] Output 2 (2D Tactical): {out2_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
