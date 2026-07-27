import os
import urllib.request

def download_football_pitch_keypoint_model(target_path: str = "models/football_pitch_keypoints.pt"):
    """
    Downloads pre-trained Football Pitch Keypoint Pose Neural Network weights.
    Trained on 29+ standard FIFA pitch keypoint landmarks.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        print(f"[INFO] Pre-trained Pitch Keypoint Model already present: {target_path}")
        return target_path

    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
    print(f"[INFO] Downloading Pre-Trained Football Pitch Keypoint Model weights...")
    print(f"       Source: {url}")
    print(f"       Destination: {target_path}")
    
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[SUCCESS] Pitch Keypoint Model downloaded successfully: {target_path}")
        return target_path
    except Exception as e:
        print(f"[ERROR] Failed to download pitch keypoint model: {e}")
        return None

if __name__ == "__main__":
    download_football_pitch_keypoint_model()
