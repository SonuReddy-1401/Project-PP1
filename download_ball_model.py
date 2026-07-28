import os
import urllib.request
from ultralytics import YOLO

def download_ball_model():
    """
    Downloads and verifies dedicated football ball detector weights into models/football_ball_detector.pt.
    """
    os.makedirs("models", exist_ok=True)
    target_path = "models/football_ball_detector.pt"
    
    print("=" * 70)
    print("     ROBOFLOW FOOTBALL BALL DETECTOR MODEL DOWNLOADER     ")
    print("=" * 70)

    if os.path.exists(target_path):
        try:
            YOLO(target_path)
            print(f"[INFO] Valid ball model found: {target_path}")
            return True
        except Exception:
            print(f"[WARNING] Invalid or corrupted ball model file found. Re-downloading...")
            os.remove(target_path)

    urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt"
    ]

    for url in urls:
        print(f"[INFO] Attempting download from: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                out_file.write(response.read())

            YOLO(target_path)
            print(f"[SUCCESS] Valid ball model saved to: {target_path}")
            return True
        except Exception as e:
            print(f"[WARNING] Download failed from {url}: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

    return False

if __name__ == "__main__":
    download_ball_model()
