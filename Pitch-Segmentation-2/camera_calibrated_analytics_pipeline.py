"""
Full Video 3D Camera Calibration & Pitch Line Segmentation Video Generator
Phase 1 - Baseline Video Renderer with Temporal Parameter Smoothing

Implements Exponential Moving Average (EMA) and Motion Velocity Extrapolation across 3D
camera parameters [Pan, Tilt, Roll, Focal Length, Position] to eliminate frame jitter
and provide smooth camera tracking through failed frames.

Author: Senior Computer Vision & ML Engineer
Workspace: Pitch-Segmentation-2
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# Add sn-calibration to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SN_CALIB_DIR = os.path.join(SCRIPT_DIR, "sn-calibration")
if SN_CALIB_DIR not in sys.path:
    sys.path.insert(0, SN_CALIB_DIR)

from src.detect_extremities import SegmentationNetwork, generate_class_synthesis, get_line_extremities
from src.baseline_cameras import normalization_transform, estimate_homography_from_line_correspondences
from src.camera import Camera, pan_tilt_roll_to_orientation, rotation_matrix_to_pan_tilt_roll
from src.soccerpitch import SoccerPitch


class TemporalCameraFilter:
    """
    Exponential Moving Average (EMA) & Velocity Extrapolation for 3D Camera Parameters.
    Filters high-frequency line jitter and smoothly predicts camera motion across failed frames.
    """
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.smooth_state = None
        self.velocity = None
        self.image_width = 960
        self.image_height = 540

    def extract_state(self, cam: Camera) -> np.ndarray:
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(cam.rotation)
        state = np.array([
            pan, tilt, roll,
            cam.xfocal_length, cam.yfocal_length,
            cam.principal_point[0], cam.principal_point[1],
            cam.position[0], cam.position[1], cam.position[2]
        ], dtype=np.float64)
        return state

    def update(self, cam: Camera) -> Camera:
        if cam is None:
            if self.smooth_state is not None and self.velocity is not None:
                # Extrapolate camera state using velocity with dampening
                self.smooth_state += self.velocity
                self.velocity *= 0.95  # Motion decay
                return self._rebuild_camera(self.smooth_state, self.image_width, self.image_height)
            return None

        raw_state = self.extract_state(cam)
        self.image_width = cam.image_width
        self.image_height = cam.image_height

        if self.smooth_state is None:
            self.smooth_state = raw_state
            self.velocity = np.zeros_like(raw_state)
        else:
            # Unwrap Euler angles to prevent 2*pi wrap-around jumps
            for i in range(3):
                diff = raw_state[i] - self.smooth_state[i]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                raw_state[i] = self.smooth_state[i] + diff

            # Update velocity and smoothed state vector
            current_velocity = raw_state - self.smooth_state
            self.velocity = self.alpha * current_velocity + (1.0 - self.alpha) * self.velocity
            self.smooth_state = self.alpha * raw_state + (1.0 - self.alpha) * (self.smooth_state + self.velocity)

        return self._rebuild_camera(self.smooth_state, self.image_width, self.image_height)

    def _rebuild_camera(self, state: np.ndarray, width: int, height: int) -> Camera:
        pan, tilt, roll = state[0], state[1], state[2]
        fx, fy = state[3], state[4]
        cx, cy = state[5], state[6]
        px, py, pz = state[7], state[8], state[9]

        cam = Camera(width, height)
        cam.rotation = pan_tilt_roll_to_orientation(pan, tilt, roll)
        cam.position = np.array([px, py, pz])
        cam.xfocal_length = fx
        cam.yfocal_length = fy
        cam.principal_point = (cx, cy)
        cam.calibration = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype='float')
        return cam


class CalibratedPitchVideoRenderer:
    def __init__(self, pitch_weights: str = None):
        print("[INFO] Initializing 3D Camera Calibration Pitch Renderer with Temporal Smoothing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load DeepLabv3 Line Extremities Model
        if pitch_weights is None:
            pitch_weights = os.path.join(SN_CALIB_DIR, "resources", "soccer_pitch_segmentation.pth")

        mean_path = os.path.join(SN_CALIB_DIR, "resources", "mean.npy")
        std_path = os.path.join(SN_CALIB_DIR, "resources", "std.npy")

        print(f"[INFO] Loading 3D Pitch Calibration Weights from: {pitch_weights}")
        self.calib_net = SegmentationNetwork(pitch_weights, mean_path, std_path)

        # 2. Load FIFA Pitch Model Definition
        self.pitch = SoccerPitch()

    def solve_frame_camera(self, frame: np.ndarray, width: int = 960, height: int = 540) -> Camera:
        """
        Segment pitch lines and estimate 3D Camera parameters for a single frame.
        """
        h_orig, w_orig = frame.shape[:2]
        semlines = self.calib_net.analyse_image(frame)
        skeletons = generate_class_synthesis(semlines, 6)
        extremities = get_line_extremities(skeletons, 40, width, height)

        if not extremities:
            return None

        # Build 3D-2D line correspondences
        line_matches = []
        potential_3d_2d_matches = {}
        src_pts = []

        for k, v in extremities.items():
            if k == 'Circle central' or "unknown" in k:
                continue
            P3D1 = self.pitch.line_extremities_keys[k][0]
            P3D2 = self.pitch.line_extremities_keys[k][1]

            p1 = np.array([v[0]['x'] * width, v[0]['y'] * height, 1.])
            p2 = np.array([v[1]['x'] * width, v[1]['y'] * height, 1.])
            src_pts.extend([p1, p2])

            for P3D in (P3D1, P3D2):
                if P3D in potential_3d_2d_matches:
                    potential_3d_2d_matches[P3D].extend([p1, p2])
                else:
                    potential_3d_2d_matches[P3D] = [p1, p2]

            line = np.cross(p1, p2)
            if np.isnan(np.sum(line)) or np.isinf(np.sum(line)):
                continue

            line_pitch = self.pitch.get_2d_homogeneous_line(k)
            if line_pitch is not None:
                line_matches.append((line_pitch, line))

        if len(line_matches) >= 4:
            target_pts = [self.pitch.point_dict[k][:2] for k in potential_3d_2d_matches.keys()]
            T1 = normalization_transform(target_pts)
            T2 = normalization_transform(src_pts)
            success, H = estimate_homography_from_line_correspondences(line_matches, T1, T2)

            if success:
                cam = Camera(width, height)
                if cam.from_homography(H):
                    point_matches = []
                    for k, potential_matches in potential_3d_2d_matches.items():
                        p3D = self.pitch.point_dict[k]
                        projected = cam.project_point(p3D)
                        if 0 < projected[0] < width and 0 < projected[1] < height:
                            dist = np.zeros(len(potential_matches))
                            for i, potential_match in enumerate(potential_matches):
                                dist[i] = np.sqrt((projected[0] - potential_match[0])**2 + (projected[1] - potential_match[1])**2)
                            selected = np.argmin(dist)
                            if dist[selected] < 100:
                                point_matches.append((p3D, potential_matches[selected][:2]))

                    if len(point_matches) > 3:
                        cam.refine_camera(point_matches)

                    if (w_orig, h_orig) != (width, height):
                        cam.scale_resolution(w_orig / width)
                    return cam

        return None

    def process_full_video(self, input_video: str, output_video: str):
        if not os.path.exists(input_video):
            print(f"[ERROR] Input video not found: {input_video}")
            return

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n[INFO] Starting Temporal Smoothed 3D Pitch Calibration Video Rendering...")
        print(f"[INFO] Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_idx = 0
        solve_success_count = 0
        solve_fail_count = 0

        cam_filter = TemporalCameraFilter(alpha=0.25)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 1. Solve raw camera parameters
            raw_cam = self.solve_frame_camera(frame)
            if raw_cam is not None:
                solve_success_count += 1
            else:
                solve_fail_count += 1

            # 2. Apply Temporal EMA & Velocity Extrapolation
            smoothed_cam = cam_filter.update(raw_cam)

            # 3. Draw Smoothed 3D Pitch Wireframe Overlay
            if smoothed_cam is not None:
                annotated_frame = smoothed_cam.draw_colorful_pitch(frame.copy(), self.pitch.palette)
            else:
                annotated_frame = frame.copy()

            writer.write(annotated_frame)

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                print(f"[PROGRESS] {frame_idx}/{total_frames} | Solved: {solve_success_count} | Failed: {solve_fail_count}")

        cap.release()
        writer.release()

        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n[SUCCESS] Temporally Smoothed 3D Pitch Calibration Video Saved!")
        print(f"[LOCATION] {output_video}")
        print(f"[FILE SIZE] {size_mb:.2f} MB")


if __name__ == "__main__":
    INPUT_VIDEO = r"C:\CLG\PP1\EXP\Pitch-Segmentation\data\input_video.mp4"
    OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "output", "pitch_calibrated_smoothed_video.mp4")

    renderer = CalibratedPitchVideoRenderer()
    renderer.process_full_video(INPUT_VIDEO, OUTPUT_VIDEO)
