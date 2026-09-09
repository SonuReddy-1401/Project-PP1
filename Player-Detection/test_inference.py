import os
import sys
import cv2
import onnxruntime as ort
from inference import get_model

print(f"Available ONNX Execution Providers: {ort.get_available_providers()}")

IMAGE_PATH = r"C:\CLG\PP1\EXP\Pitch-Calibration-Clean\data\custom_dataset\test\clip1\frame_00039.jpg"

api_key = os.environ.get("ROBOFLOW_API_KEY")
if len(sys.argv) > 1:
    api_key = sys.argv[1]

if not api_key:
    print("\nERROR: Roboflow API Key not provided.")
    print("Please run: python test_inference.py <YOUR_ROBOFLOW_API_KEY>")
    sys.exit(1)

print("\nLoading model 'football-players-detection-3zvbc/1' via Roboflow inference package...")
model = get_model(model_id="football-players-detection-3zvbc/1", api_key=api_key)

print(f"Reading frame: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"ERROR: Could not load image from {IMAGE_PATH}")
    sys.exit(1)

print("Running inference...")
results = model.infer(image)[0]

print(f"\n--- Detection Results for frame_00039.jpg ---")
print(f"Total Detections: {len(results.predictions)}")
for i, pred in enumerate(results.predictions, 1):
    print(f"{i:2d}. Class: {pred.class_name:<12} Confidence: {pred.confidence:.4f} | Center: ({pred.x:6.1f}, {pred.y:6.1f}) | Size: {pred.width:5.1f}x{pred.height:5.1f}")

cache_dir = os.path.expanduser(r"~\.cache\inference")
print(f"\nLocal Cache Location: {cache_dir}")
if os.path.exists(cache_dir):
    print(f"Local Cache Contents: {os.listdir(cache_dir)}")
