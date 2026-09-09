"""
Screen-Overlay vs Pitch-Painted Check
Detects candidate "logo" false-positive boxes near the frame corners across
multiple frames, prints their exact coordinates for comparison, and draws them
onto each frame so you can visually confirm consistent position.
"""
import os
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

VIDEO_PATH = r"C:\CLG\PP1\EXP\PnLCalib\data\broadcast_clip_1.mp4"
OUTPUT_DIR = r"C:\CLG\PP1\EXP\Player-Detection\logo_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same 10 frame indices from the earlier robustness test (seed 42)
FRAME_INDICES = [204, 839, 912, 1143, 1828, 2006, 2253, 5238, 6033, 6074]

yolo_path = hf_hub_download(repo_id="uisikdag/yolo-v8-football-players-detection", filename="best.pt")
model = YOLO(yolo_path)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Frame size: {frame_w}x{frame_h}\n")

found_any = False

for idx in FRAME_INDICES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        print(f"[WARNING] Could not read frame {idx}")
        continue

    results = model.predict(frame, device=0, verbose=False)
    annotated = frame.copy()
    logo_boxes_this_frame = []

    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Broad corner-region flag: any edge/corner region, not assuming which corner
            near_left = x1 < frame_w * 0.15
            near_right = x2 > frame_w * 0.85
            near_top = y1 < frame_h * 0.20
            near_bottom = y2 > frame_h * 0.85
            is_corner_region = (near_left or near_right) and (near_top or near_bottom)

            if is_corner_region:
                logo_boxes_this_frame.append((cls_name, conf, x1, y1, x2, y2))
                found_any = True
                print(f"Frame {idx}: CORNER-REGION detection -> {cls_name} {conf:.2f} "
                      f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                # Draw in bright magenta so it's unmistakable in the saved image
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                cv2.putText(annotated, f"CORNER: {cls_name} {conf:.2f}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if logo_boxes_this_frame:
        out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:05d}_logo_check.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"  -> Saved: {out_path}\n")
    else:
        print(f"Frame {idx}: no corner-region detections found\n")

cap.release()

if not found_any:
    print("[RESULT] No corner-region detections found in any of the 10 frames.")
else:
    print("[RESULT] Review the printed coordinates above and the saved images in:")
    print(f"  {OUTPUT_DIR}")
    print("If the (x1,y1,x2,y2) values are near-identical across frames where it appears,")
    print("this confirms a fixed screen-space overlay (simple rectangle exclusion fixes it).")
