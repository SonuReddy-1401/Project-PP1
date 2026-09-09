"""
Visual Verification Script for SoccerNet 3D Camera Calibration
Phase 1 - Baseline Assessment

Reads camera_XXXXX.json outputs, loads 3D Camera parameters, and projects
canonical 3D FIFA pitch lines onto original video frames to assess calibration accuracy.

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation-2
"""

import os
import sys
import json
import glob
import cv2
import numpy as np

# Add sn-calibration directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SN_CALIB_DIR = os.path.join(SCRIPT_DIR, "sn-calibration")
sys.path.append(SN_CALIB_DIR)

try:
    from src.camera import Camera
    from src.soccerpitch import SoccerPitch
except ImportError as e:
    print(f"[ERROR] Could not import sn-calibration modules: {e}")
    sys.exit(1)


def draw_pitch_3d_overlay(image: np.ndarray, camera: Camera, pitch: SoccerPitch) -> np.ndarray:
    """
    Project canonical 3D pitch lines onto 2D image plane using estimated Camera.
    """
    overlay = image.copy()
    w, h = camera.width, camera.height

    # Loop through standard FIFA pitch line definitions
    for line_class, line_key in pitch.line_extremities_keys.items():
        if line_class in pitch.palette:
            color = pitch.palette[line_class]
            # Convert palette color to BGR for OpenCV
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
        else:
            bgr_color = (0, 255, 0)  # Default green

        pt3D_1_key = line_key[0]
        pt3D_2_key = line_key[1]

        if pt3D_1_key in pitch.point_dict and pt3D_2_key in pitch.point_dict:
            pt3D_1 = pitch.point_dict[pt3D_1_key]
            pt3D_2 = pitch.point_dict[pt3D_2_key]

            # Project 3D world coordinates (X, Y, Z) to 2D image coordinates (x, y)
            pt2D_1 = camera.project_point(pt3D_1)
            pt2D_2 = camera.project_point(pt3D_2)

            # Check if points are valid numbers and inside/near screen boundary
            if not np.isnan(pt2D_1).any() and not np.isnan(pt2D_2).any():
                x1, y1 = int(pt2D_1[0]), int(pt2D_1[1])
                x2, y2 = int(pt2D_2[0]), int(pt2D_2[1])

                # Draw line segment
                cv2.line(overlay, (x1, y1), (x2, y2), bgr_color, 2, cv2.LINE_AA)

    # Blend overlay with original image
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    return result


def visualize_all_calibrations(
    frames_dir: str,
    predictions_dir: str,
    output_dir: str,
    resolution_width: int = 960,
    resolution_height: int = 540
):
    os.makedirs(output_dir, exist_ok=True)
    pitch = SoccerPitch()

    camera_files = sorted(glob.glob(os.path.join(predictions_dir, "**", "camera_*", "*.json"), recursive=True))
    if not camera_files:
        camera_files = sorted(glob.glob(os.path.join(predictions_dir, "**", "camera_*.json"), recursive=True))
    if not camera_files:
        camera_files = [f for f in glob.glob(os.path.join(predictions_dir, "**", "*.json"), recursive=True) if "extremities" not in f]
    if not camera_files:
        print(f"[WARNING] No camera JSON files found in: {predictions_dir}")
        return

    print(f"[INFO] Found {len(camera_files)} camera parameter JSON files.")
    success_count = 0

    for cam_file in camera_files:
        filename = os.path.basename(cam_file)
        clean_idx = filename.replace("camera_", "").replace("frame_", "").replace(".json", "")
        target_name = f"frame_{clean_idx}.jpg"

        frame_file = os.path.join(frames_dir, target_name)
        if not os.path.exists(frame_file):
            matching_frames = glob.glob(os.path.join(frames_dir, "**", target_name), recursive=True)
            if matching_frames:
                frame_file = matching_frames[0]
            else:
                print(f"[WARNING] Original frame image missing for: {filename}")
                continue

        image = cv2.imread(frame_file)
        if image is None:
            continue

        h_orig, w_orig = image.shape[:2]

        with open(cam_file, "r") as f:
            cam_params = json.load(f)

        if not cam_params:
            print(f"[SKIP] Empty camera parameters in: {filename}")
            continue

        camera = Camera(resolution_width, resolution_height)
        camera.from_json_parameters(cam_params)

        # Scale camera parameters if frame resolution differs from model resolution
        if (w_orig, h_orig) != (resolution_width, resolution_height):
            scale_factor = w_orig / resolution_width
            camera.scale_resolution(scale_factor)

        overlay_img = camera.draw_colorful_pitch(image, pitch.palette)

        out_path = os.path.join(output_dir, f"overlay_{clean_idx}.jpg")
        cv2.imwrite(out_path, overlay_img)
        success_count += 1

    print(f"\n[SUCCESS] Generated {success_count} visual calibration overlays in:")
    print(f" -> {output_dir}")


if __name__ == "__main__":
    FRAMES_DIR = os.path.join(SCRIPT_DIR, "data", "custom_dataset", "test", "clip1")
    PRED_DIR = os.path.join(SCRIPT_DIR, "output", "predictions", "test")
    OUT_DIR = os.path.join(SCRIPT_DIR, "output", "visual_verification")

    visualize_all_calibrations(
        frames_dir=FRAMES_DIR,
        predictions_dir=PRED_DIR,
        output_dir=OUT_DIR
    )
