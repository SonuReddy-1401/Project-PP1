import time
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

cuda_available = torch.cuda.is_available()
device = 0 if cuda_available else "cpu"

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available in PyTorch: {cuda_available}")
if cuda_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU mode for fallback test...")

repo_id = "uisikdag/yolo-v8-football-players-detection"
filename = "best.pt"

print(f"\nDownloading '{filename}' from Hugging Face repo '{repo_id}'...")
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
print(f"Downloaded model weight path:\n{model_path}")

print("\nLoading model with standard ultralytics YOLO...")
model = YOLO(model_path)

print(f"\nModel Class Names: {model.names}")

IMAGE_PATH = r"C:\CLG\PP1\EXP\Pitch-Calibration-Clean\data\custom_dataset\test\clip1\frame_00039.jpg"
print(f"\nRunning inference on {IMAGE_PATH} with device='{device}'...")

start_time = time.time()
results = model.predict(source=IMAGE_PATH, device=device, verbose=False)
elapsed_ms = (time.time() - start_time) * 1000

print(f"\nInference completed in {elapsed_ms:.2f} ms")

print("\n--- Detection Results for frame_00039.jpg ---")
total_boxes = 0
for r in results:
    for box in r.boxes:
        total_boxes += 1
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        print(f"{total_boxes:2d}. Class: {cls_name:<12} Conf: {conf:.4f} | BBox [x1, y1, x2, y2]: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

print(f"\nTotal Detected Objects: {total_boxes}")
