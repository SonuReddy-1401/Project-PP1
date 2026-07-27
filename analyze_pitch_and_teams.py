import os
import cv2
import numpy as np
from ultralytics import YOLO

def analyze_frame_0(frame_path: str = "data/Temp/frame_0.jpg"):
    if not os.path.exists(frame_path):
        print(f"[ERROR] Frame file not found: {frame_path}")
        return

    frame = cv2.imread(frame_path)
    h, w, _ = frame.shape
    print(f"[INFO] Loaded frame 0 ({w}x{h})")

    # Run YOLOv8x on Frame 0
    model = YOLO("models/yolov8x.pt")
    results = model(frame, conf=0.10, verbose=False)[0]
    
    person_crops = []
    red_ratios = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                h_crop = y2 - y1
                torso = crop[int(h_crop*0.15):int(h_crop*0.45), :]
                if torso.size > 0:
                    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                    m1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
                    m2 = cv2.inRange(hsv, np.array([155, 50, 40]), np.array([180, 255, 255]))
                    red_mask = cv2.bitwise_or(m1, m2)
                    r_ratio = np.sum(red_mask > 0) / float(torso.shape[0] * torso.shape[1] + 1e-5)
                    red_ratios.append(r_ratio)
                    print(f"Player BBox [{x1}, {y1}, {x2}, {y2}] -> Red Jersey Ratio: {r_ratio:.2f}")

    # Pitch Grass Contour detection
    hsv_f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv_f, np.array([30, 25, 25]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c_max)
        box_pts = cv2.boxPoints(rect)
        print("\n[PITCH GRASS BOUNDING BOX CORNERS (Pixels)]:")
        for pt in box_pts:
            print(f"  Pixel: ({pt[0]:.1f}, {pt[1]:.1f})")

if __name__ == "__main__":
    analyze_frame_0()
