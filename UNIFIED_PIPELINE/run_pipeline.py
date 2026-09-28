#!/usr/bin/env python3
"""
Unified Production Flexible Football Analytics Pipeline
One-click CLI entrypoint for processing arbitrary video clips / position datasets
and auto-launching the Pro Match Command Center interactive web dashboard.
"""
import os
import sys
import json
import argparse
import webbrowser
import http.server
import socketserver
import threading
import time

# Reconfigure stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure pipeline package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipeline.processor import process_dataset
from pipeline.exporter import export_pipeline_assets
from pipeline.cv_engine import process_video_clip

def run_http_server(directory, port):
    """
    Launches background HTTP server serving dashboard directory.
    """
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"🚀 Dashboard HTTP Server active at http://localhost:{port}/")
        httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(
        description="Unified Production Flexible Football Analytics Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default=None, help="Path to input positions JSON dataset file")
    parser.add_argument("--video", type=str, default=None, help="Path to raw broadcast video file (.mp4, .avi, .mov)")
    parser.add_argument("--team_name", type=str, default="SSC Napoli", help="Target team/squad label")
    parser.add_argument("--opponent_name", type=str, default="AS Roma", help="Opponent team label")
    parser.add_argument("--team_color", type=str, default="sky_blue", help="Target kit color theme (sky_blue, red, white, yellow, emerald, navy, or hex code #0080FF)")
    parser.add_argument("--match_start", type=str, default="09:54", help="Starting match clock timestamp (e.g., '09:54', '00:00', '45:00')")
    parser.add_argument("--fps", type=float, default=25.0, help="Video frame rate FPS")
    parser.add_argument("--port", type=int, default=8090, help="HTTP server port for dashboard")
    parser.add_argument("--no_launch", action="store_true", help="Disable automatic browser opening")

    args = parser.parse_args()

    dashboard_dir = os.path.join(current_dir, "dashboard")

    # 1. Load / Extract Position Data from Input Video or Dataset
    input_data = []
    fps = args.fps

    if args.video and os.path.exists(args.video):
        print(f"🎬 Processing broadcast video clip: {args.video}")
        input_data, video_fps, total_v_frames, duration_v_sec = process_video_clip(
            video_path=args.video,
            kit_color_input=args.team_color,
            target_fps=args.fps
        )
        fps = video_fps
    elif args.dataset and os.path.exists(args.dataset):
        print(f"📂 Loading input dataset: {args.dataset}")
        with open(args.dataset, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    else:
        # Fallback to default clip dataset if present
        default_ds = os.path.join(current_dir, "..", "PRO_DASHBOARD", "data", "positions_dataset_clip15min.json")
        if os.path.exists(default_ds):
            print(f"📂 Using default dataset: {default_ds}")
            with open(default_ds, "r", encoding="utf-8") as f:
                input_data = json.load(f)
        else:
            print("⚠️ No valid input dataset provided and default dataset not found!")
            sys.exit(1)

    print(f"⚡ Processing {len(input_data)} frames with FPS={args.fps} and match start clock={args.match_start}...")
    processed_frames, stats = process_dataset(input_data, fps=args.fps)

    print(f"✅ Metrics calculated:")
    print(f"   • Average Pitch Width: {stats['avg_width_m']} m")
    print(f"   • Average Pitch Depth: {stats['avg_depth_m']} m")
    print(f"   • Average Squad Area: {stats['avg_area_m2']} m²")
    print(f"   • Middle Third Occupancy: {stats['thirds_pct']['middle']}%")
    print(f"   • Safe Centroid Distance: {stats['safe_centroid_dist_km']} km ({stats['avg_pace_kmh']} km/h avg pace)")

    # 2. Export Pipeline Assets
    print("📦 Exporting pipeline assets to dashboard/data/...")
    ds_path, meta_path, nar_path = export_pipeline_assets(
        dashboard_dir=dashboard_dir,
        processed_frames=processed_frames,
        stats=stats,
        team_name=args.team_name,
        opponent_name=args.opponent_name,
        team_color=args.team_color,
        match_start_clock=args.match_start,
        fps=args.fps
    )

    # 3. Launch HTTP Server Thread
    server_thread = threading.Thread(target=run_http_server, args=(dashboard_dir, args.port), daemon=True)
    server_thread.start()

    time.sleep(1.0)
    url = f"http://localhost:{args.port}/"
    print(f"🌐 Pro Tactical Command Center ready at: {url}")

    if not args.no_launch:
        print("🖥️ Opening browser...")
        webbrowser.open(url)

    print("Press Ctrl+C to terminate the dashboard server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Dashboard server terminated gracefully.")

if __name__ == "__main__":
    main()
