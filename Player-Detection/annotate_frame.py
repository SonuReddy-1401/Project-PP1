import os
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

repo_id = "uisikdag/yolo-v8-football-players-detection"
filename = "best.pt"
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
model = YOLO(model_path)

IMAGE_PATH = r"C:\CLG\PP1\EXP\Pitch-Calibration-Clean\data\custom_dataset\test\clip1\frame_00039.jpg"
OUTPUT_PATH = r"C:\CLG\PP1\EXP\Player-Detection\frame_00039_annotated.jpg"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

results = model.predict(source=IMAGE_PATH, device=0, verbose=False)

color_map = {
    "player": (0, 255, 0),       # Green
    "goalkeeper": (255, 255, 0), # Cyan/Yellow
    "referee": (255, 0, 0),      # Blue
    "ball": (0, 0, 255),         # Red
}

for r in results:
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = color_map.get(cls_name, (0, 255, 255))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite(OUTPUT_PATH, image)
print(f"Saved annotated image to: {OUTPUT_PATH}")
