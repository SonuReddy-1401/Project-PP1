import os
import subprocess
import imageio_ffmpeg

def download_youtube_clip(url: str, start_time: str, end_time: str, output_path: str):
    """
    Downloads a specific timestamp section from a YouTube video in high resolution (1080p)
    using yt-dlp and imageio-ffmpeg.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    cmd = [
        ".\\.venv\\Scripts\\yt-dlp.exe",
        "--ffmpeg-location", ffmpeg_exe,
        "--download-sections", f"*{start_time}-{end_time}",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--force-overwrites",
        "-o", output_path,
        url
    ]
    
    print(f"[INFO] Downloading broadcast clip: {start_time} -> {end_time}")
    print(f"       Target file: {output_path}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SUCCESS] Download completed: {output_path}")
    else:
        print(f"[ERROR] Download failed: {res.stderr}")

def main():
    yt_url = "https://www.youtube.com/live/EGjiKT12JR8"
    
    # Clip 1: 00:45:17 to 00:46:10 (53s)
    clip1_path = "data/input/broadcast_clip_1.mp4"
    download_youtube_clip(yt_url, "00:45:17", "00:46:10", clip1_path)
    
    # Clip 2: 01:09:50 to 01:11:06 (1m 16s)
    clip2_path = "data/input/broadcast_clip_2.mp4"
    download_youtube_clip(yt_url, "01:09:50", "01:11:06", clip2_path)

if __name__ == "__main__":
    main()
