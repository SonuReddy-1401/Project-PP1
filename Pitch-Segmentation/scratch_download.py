import os
import subprocess
import yt_dlp

url = "https://www.youtube.com/watch?v=RV27Xqg3MxM"
temp_video = r"C:\CLG\PP1\EXP\Pitch-Segmentation\data\temp_1080p_full.mp4"
output_file = r"C:\CLG\PP1\EXP\Pitch-Segmentation\data\input_video_clip_23_24_1080p.mp4"

print(f"[INFO] Step 1: Downloading 1080p Full Video Stream via native yt-dlp...")

ydl_opts = {
    'format': '137/bestvideo[height>=1080]',
    'outtmpl': temp_video,
    'overwrites': True,
    'quiet': False
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

if not os.path.exists(temp_video):
    print("[ERROR] 1080p video download failed.")
    exit(1)

print(f"\n[INFO] Step 2: Trimming 1-minute clip (23:00 to 24:00) using FFmpeg...")

cmd = [
    'ffmpeg',
    '-ss', '00:23:00',
    '-i', temp_video,
    '-t', '00:01:00',
    '-c', 'copy',
    output_file,
    '-y'
]

res = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(temp_video):
    os.remove(temp_video)

if res.returncode == 0 and os.path.exists(output_file):
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n[SUCCESS] 1080p Full Quality 1-minute clip created successfully!")
    print(f"[RESOLUTION] 1920x1080")
    print(f"[FILE SIZE] {size_mb:.2f} MB")
    print(f"[LOCATION] {output_file}")
else:
    print(f"\n[ERROR] FFmpeg trim failed: {res.returncode}")
    print("Stderr:", res.stderr[:500])
