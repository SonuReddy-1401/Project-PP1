import os
import sys
import subprocess
# pyrefly: ignore [missing-import]
import imageio_ffmpeg
from main import run_pipeline

def download_youtube_clip(url: str, start_time: str, end_time: str, output_path: str):
    """
    Downloads a specific timestamp section of a YouTube video using yt-dlp & imageio-ffmpeg.
    start_time & end_time format: HH:MM:SS or MM:SS (e.g. '00:00:00', '00:02:30')
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    print("=" * 70)
    print(f"[INFO] DOWNLOADING YOUTUBE CLIP: {url}")
    print(f"       Time Segment: {start_time} --> {end_time}")
    print(f"       Output Path:  {output_path}")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--ffmpeg-location", ffmpeg_path,
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        url
    ]
    
    subprocess.run(cmd, check=True)
    print(f"[SUCCESS] Download completed: {output_path}\n")

def process_and_analyze_clips():
    yt_url = "https://youtu.be/9x02ovOrZmM"
    
    clip1_input = "data/input/tactical_clip_1.mp4"
    clip1_output_video = "data/output/tactical_clip_1_annotated.mp4"
    clip1_output_csv = "data/output/tactical_clip_1_metrics.csv"
    
    clip2_input = "data/input/tactical_clip_2.mp4"
    clip2_output_video = "data/output/tactical_clip_2_annotated.mp4"
    clip2_output_csv = "data/output/tactical_clip_2_metrics.csv"

    # 1. Download Clip 1 (00:00:00 to 00:02:30 - 2 minutes 30 seconds)
    if not os.path.exists(clip1_input):
        download_youtube_clip(yt_url, "00:02:30", "00:05:00", clip1_input)
    else:
        print(f"[INFO] Using existing Clip 1: {clip1_input}")

    # 2. Download Clip 2 (00:02:30 to 00:05:00 - 2 minutes 30 seconds)
    if not os.path.exists(clip2_input):
        download_youtube_clip(yt_url, "00:10:00", "00:12:30", clip2_input)
    else:
        print(f"[INFO] Using existing Clip 2: {clip2_input}")

    # 3. Analyze Clip 1
    print("\n" + "=" * 70)
    print("      ANALYZING CLIP 1 (00:00 - 02:30)      ")
    print("=" * 70)
    run_pipeline(
        input_video=clip1_input, 
        output_video=clip1_output_video, 
        output_csv=clip1_output_csv,
        model_path="yolov8m.pt",
        imgsz=1280,
        conf_threshold=0.15,
        frame_stride=2
    )

    # 4. Analyze Clip 2
    print("\n" + "=" * 70)
    print("      ANALYZING CLIP 2 (02:30 - 05:00)      ")
    print("=" * 70)
    run_pipeline(
        input_video=clip2_input, 
        output_video=clip2_output_video, 
        output_csv=clip2_output_csv,
        model_path="yolov8m.pt",
        imgsz=1280,
        conf_threshold=0.15,
        frame_stride=2
    )

    print("\n[FINISHED] Both 2:30 clips processed successfully!")
    print(f"Clip 1 Video: {clip1_output_video} | CSV: {clip1_output_csv}")
    print(f"Clip 2 Video: {clip2_output_video} | CSV: {clip2_output_csv}")

if __name__ == "__main__":
    process_and_analyze_clips()
