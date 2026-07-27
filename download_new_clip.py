import os
import sys
import subprocess

def download_youtube_clip_1080p(url: str, start_time: str, end_time: str, output_path: str):
    """
    Downloads Full 1080p HD video section from YouTube using yt-dlp and imageio_ffmpeg.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print("=" * 70)
    print("      YOUTUBE BROADCAST 1080P FULL HD CLIP DOWNLOADER      ")
    print("=" * 70)
    print(f"[INFO] Source URL: {url}")
    print(f"[INFO] Section Timestamp: {start_time} -> {end_time}")
    print(f"[INFO] Target Output File: {output_path}")

    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    python_exe = sys.executable

    # Force format 137+140 for 1920x1080 Full HD
    cmd = [
        python_exe, "-m", "yt_dlp",
        "--download-sections", f"*{start_time}-{end_time}",
        "-f", "137+140/bestvideo[height>=1080]+bestaudio/best",
        "--merge-output-format", "mp4",
        "--ffmpeg-location", ffmpeg_exe,
        "-o", output_path,
        "--force-overwrites",
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] 1080p Full HD Download completed: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download 1080p clip: {e}")
        return False

if __name__ == "__main__":
    yt_url = "https://youtu.be/86zhlXNNUZI"
    out_file = "data/input/new_match_red_team_1080p.mp4"
    download_youtube_clip_1080p(yt_url, "00:26:00", "00:27:00", out_file)
