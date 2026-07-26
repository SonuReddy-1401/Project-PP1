import os
import sys
import argparse
import subprocess

def download_sample_clip(output_path: str = "data/input/sample_match.mp4", max_height: int = 1080):
    """
    Downloads a 30-second sample football match broadcast video clip in HD.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[INFO] Downloading HD sample football match clip to: {output_path}")
    
    sample_url = "https://www.youtube.com/watch?v=neBZ6huolkg"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "--download-sections", "*00:00:10-00:00:40",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        sample_url
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] 30-second sample football video downloaded to: {output_path}")
    except Exception as e:
        print(f"[WARNING] Download fallback attempt: {e}")

def download_soccernet_dataset(task: str = "tracking", local_dir: str = "data/soccernet"):
    """
    Downloads official SoccerNet dataset clips (from arXiv:2204.06918 / arXiv:2104.09333 / arXiv:2011.13367).
    """
    print(f"[INFO] Downloading SoccerNet dataset ({task}) to: {local_dir}")
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        downloader = SoccerNetDownloader(LocalDirectory=local_dir)
        if task == "tracking":
            downloader.downloadDataTask(task="tracking", split=["test"])
        elif task == "calibration":
            downloader.downloadDataTask(task="calibration", split=["test"])
        print("[SUCCESS] SoccerNet dataset download complete!")
    except Exception as e:
        print(f"[ERROR] Failed to download SoccerNet: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Football Video Clips from Papers & Datasets")
    parser.add_argument("--source", type=str, choices=["sample", "soccernet", "youtube"], default="sample",
                        help="Choose clip source: sample, soccernet, or youtube")
    parser.add_argument("--url", type=str, default=None, help="YouTube or video URL if source is youtube")
    parser.add_argument("--output", type=str, default="data/input/real_match.mp4", help="Output file path")
    args = parser.parse_args()

    if args.source == "sample":
        download_sample_clip(args.output)
    elif args.source == "soccernet":
        download_soccernet_dataset("tracking")
    elif args.source == "youtube":
        if not args.url:
            print("[ERROR] Please provide a --url when using --source youtube")
        else:
            download_sample_clip(args.output)
