"""
Pitch-Boundary Detection Filter — PnLCalib version
Same logic as the sn-calibration version, adapted for PnLCalib's exported JSON
structure (nested "cam_params" dict, 1920x1080-native principal point).
"""
import os
import sys
import json
import cv2
import numpy as np

SN_CALIB_DIR = r"C:\CLG\PP1\EXP\Pitch-Calibration-Clean\sn-calibration"
sys.path.append(SN_CALIB_DIR)

from src.camera import Camera
from src.soccerpitch import SoccerPitch

FRAME_PATH = r"C:\CLG\PP1\EXP\PnLCalib\data\frame_00039.jpg"
CAMERA_JSON_PATH = r"C:\CLG\PP1\EXP\PnLCalib\output\pnlcalib_camera_clip1min\frame_00039.json"
OUTPUT_PATH = r"C:\CLG\PP1\EXP\Player-Detection\frame_00039_filtered_pnlcalib.jpg"

DETECTIONS = [
    ("player", 0.9077, 1101.7, 831.8, 1137.3, 909.1),
    ("player", 0.9062, 606.1, 659.1, 628.0, 721.8),
    ("player", 0.8966, 168.3, 764.4, 197.3, 837.6),
    ("player", 0.8910, 164.9, 627.9, 186.9, 692.4),
    ("player", 0.8880, 1680.4, 580.6, 1703.7, 642.4),
    ("player", 0.8873, 1292.9, 537.7, 1319.4, 595.2),
    ("player", 0.8843, 480.3, 400.4, 499.8, 449.4),
    ("player", 0.8826, 106.7, 654.5, 132.7, 714.7),
    ("player", 0.8798, 744.1, 359.5, 762.2, 401.0),
    ("player", 0.8758, 259.6, 398.9, 281.1, 444.4),
    ("player", 0.8734, 1047.7, 545.9, 1067.2, 603.7),
    ("player", 0.8684, 302.6, 819.1, 323.2, 892.6),
    ("player", 0.8681, 225.7, 452.7, 248.9, 507.3),
    ("player", 0.8621, 430.7, 644.0, 457.3, 708.2),
    ("player", 0.8599, 1867.0, 799.1, 1897.2, 869.9),
    ("player", 0.8519, 1384.6, 390.6, 1402.8, 439.0),
    ("referee", 0.8464, 792.3, 524.0, 806.9, 593.2),
    ("player", 0.8407, 1181.4, 927.0, 1237.9, 987.0),
    ("player", 0.8397, 907.2, 598.2, 934.3, 657.0),
    ("player", 0.8343, 577.3, 486.5, 596.6, 539.1),
    ("player", 0.8039, 1087.0, 411.6, 1101.4, 460.1),
    ("referee", 0.7951, 250.3, 336.9, 271.5, 378.4),
    ("player", 0.7074, 911.1, 750.4, 929.9, 820.0),
    ("referee", 0.7057, 610.3, 460.5, 633.7, 511.4),
    ("player", 0.6557, 1639.5, 488.5, 1659.8, 543.5),
    ("referee", 0.5919, 824.6, 307.8, 842.6, 349.0),
    ("ball", 0.3864, 1602.2, 624.7, 1612.5, 635.5),
    ("referee", 0.3340, 1510.5, 300.1, 1524.0, 332.2),
    ("goalkeeper", 0.2658, 1874.7, 477.2, 1895.3, 524.1),
]


def main():
    pitch = SoccerPitch()

    with open(CAMERA_JSON_PATH, "r") as f:
        data = json.load(f)

    cam_params = data["cam_params"]  # PnLCalib nests params one level deeper
    print(f"[INFO] PnLCalib mode: {data.get('mode')} | rep_err: {data.get('rep_err'):.3f}")

    calib_w = round(cam_params["principal_point"][0] * 2)   # ~1920
    calib_h = round(cam_params["principal_point"][1] * 2)   # ~1080
    print(f"[INFO] Inferred calibration resolution: {calib_w}x{calib_h}")

    image = cv2.imread(FRAME_PATH)
    if image is None:
        print(f"[ERROR] Could not load frame at {FRAME_PATH} — extract it first if missing.")
        return
    h_orig, w_orig = image.shape[:2]

    camera = Camera(calib_w, calib_h)
    camera.from_json_parameters(cam_params)
    if (w_orig, h_orig) != (calib_w, calib_h):
        scale_factor = w_orig / calib_w
        camera.scale_resolution(scale_factor)
        print(f"[INFO] Applied scale factor: {scale_factor}")

    corner_keys = ["TL_PITCH_CORNER", "TR_PITCH_CORNER", "BR_PITCH_CORNER", "BL_PITCH_CORNER"]
    polygon_points = []
    for key in corner_keys:
        pt3d = pitch.point_dict[key]
        pt2d = camera.project_point(pt3d)
        if np.isnan(pt2d).any():
            print(f"[WARNING] Corner {key} projected to NaN")
            continue
        polygon_points.append((float(pt2d[0]), float(pt2d[1])))

    print("[INFO] Projected pitch boundary polygon (image coords):")
    for key, pt in zip(corner_keys, polygon_points):
        print(f"  {key}: {pt}")

    polygon_np = np.array(polygon_points, dtype=np.float32)

    kept, rejected = [], []
    for cls_name, conf, x1, y1, x2, y2 in DETECTIONS:
        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        result = cv2.pointPolygonTest(polygon_np, (foot_x, foot_y), False)
        (kept if result >= 0 else rejected).append((cls_name, conf, x1, y1, x2, y2))

    print(f"\n[RESULT] Kept (on-pitch): {len(kept)} / {len(DETECTIONS)}")
    print(f"[RESULT] Rejected (off-pitch): {len(rejected)} / {len(DETECTIONS)}")
    for cls_name, conf, x1, y1, x2, y2 in rejected:
        print(f"  REJECTED: {cls_name} {conf:.2f} at foot=({(x1+x2)/2:.0f}, {y2:.0f})")

    poly_int = polygon_np.astype(int)
    cv2.polylines(image, [poly_int], isClosed=True, color=(0, 255, 255), thickness=3)

    for cls_name, conf, x1, y1, x2, y2 in kept:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0) if cls_name == "player" else (0, 0, 255) if cls_name == "ball" else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for cls_name, conf, x1, y1, x2, y2 in rejected:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(image, "REJECTED", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    cv2.imwrite(OUTPUT_PATH, image)
    print(f"\n[SUCCESS] Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
