"""
Generate 3D Pitch Wireframe Overlay JPEGs for clip2.
Phase 1 Clean Rebuild - Stage 3a (clip2)
"""
import os
import sys
import glob
import json
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SN_CALIB_DIR = os.path.join(SCRIPT_DIR, "sn-calibration")
sys.path.append(SN_CALIB_DIR)

from src.camera import Camera
from src.soccerpitch import SoccerPitch

FRAMES_DIR = os.path.join(SCRIPT_DIR, "data", "custom_dataset", "test", "clip2")
CAMERA_DIR = os.path.join(SCRIPT_DIR, "output", "predictions", "test", "camera_clip2")
OUT_DIR = os.path.join(SCRIPT_DIR, "output", "visual_verification_clip2")
CALIB_RESOLUTION_W, CALIB_RESOLUTION_H = 960, 540

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pitch = SoccerPitch()

    camera_files = sorted(glob.glob(os.path.join(CAMERA_DIR, "*.json")))
    if not camera_files:
        print(f"[WARNING] No camera JSON files found in: {CAMERA_DIR}")
        return

    print(f"[INFO] Found {len(camera_files)} camera parameter JSON files for clip2.")
    success_count = 0

    for cam_file in camera_files:
        filename = os.path.basename(cam_file)
        frame_name = filename.replace("camera_", "").replace(".json", ".jpg")
        frame_path = os.path.join(FRAMES_DIR, frame_name)

        if not os.path.exists(frame_path):
            continue

        image = cv2.imread(frame_path)
        if image is None:
            continue

        h_orig, w_orig = image.shape[:2]

        with open(cam_file, "r") as f:
            cam_params = json.load(f)

        if not cam_params:
            continue

        camera = Camera(CALIB_RESOLUTION_W, CALIB_RESOLUTION_H)
        camera.from_json_parameters(cam_params)

        if (w_orig, h_orig) != (CALIB_RESOLUTION_W, CALIB_RESOLUTION_H):
            scale_factor = w_orig / CALIB_RESOLUTION_W
            camera.scale_resolution(scale_factor)

        overlay_img = camera.draw_colorful_pitch(image, pitch.palette)

        out_path = os.path.join(OUT_DIR, f"overlay_{frame_name}")
        cv2.imwrite(out_path, overlay_img)
        success_count += 1

    print(f"\n[SUCCESS] Generated {success_count} visual calibration overlays in:")
    print(f" -> {OUT_DIR}")

if __name__ == "__main__":
    main()
