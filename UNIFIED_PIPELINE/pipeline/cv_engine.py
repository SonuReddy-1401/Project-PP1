"""
Production Computer Vision Feature Extraction Engine
Integrates REAL proven components from the existing codebase:
  - Stage A: Bootstrap Team Color Discovery (circular-hue K-Means k=2)
  - Stage B: PnLCalib HRNet Camera Calibration (per-frame inference)
  - Stage C: YOLO Football Detection + Torso HSV Classification + Pitch Polygon Filtering
  - Stage D: IoU Temporal Smoothing (N=5 rolling window)

References:
  - S:\\CLG\\PP1\\EXP\\Player-Detection\\bootstrap_team_colors.py
  - S:\\CLG\\PP1\\EXP\\Player-Detection\\team_classifier.py
  - S:\\CLG\\PP1\\EXP\\5-PnLCalib\\inference.py
  - S:\\CLG\\PP1\\EXP\\NEXT\\scripts\\02_run_phase_B_benchmark.py
"""
import os
import sys
import math
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path Setup — import from existing proven modules
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_DIR = os.path.dirname(CURRENT_DIR)
EXP_DIR = os.path.dirname(UNIFIED_DIR)

PLAYER_DETECTION_DIR = os.path.join(EXP_DIR, "Player-Detection")
if PLAYER_DETECTION_DIR not in sys.path:
    sys.path.insert(0, PLAYER_DETECTION_DIR)

PNLCALIB_DIR = os.path.join(EXP_DIR, "5-PnLCalib")
if not os.path.exists(PNLCALIB_DIR):
    PNLCALIB_DIR = os.path.join(EXP_DIR, "PnLCalib")
if PNLCALIB_DIR not in sys.path:
    sys.path.insert(0, PNLCALIB_DIR)

# Imports from existing proven modules
from team_classifier import get_torso_crop, extract_dominant_hsv, circular_hue_distance

# ---------------------------------------------------------------------------
# Global model caches
# ---------------------------------------------------------------------------
_YOLO_MODEL = None
_PNLCALIB_MODELS = None  # (model_kp, model_line, cam_calib)


def get_yolo_model():
    """Load fine-tuned football YOLOv8 detector (same as 02_run_phase_B_benchmark.py)."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        yolo_path = hf_hub_download(
            repo_id="uisikdag/yolo-v8-football-players-detection",
            filename="best.pt"
        )
        _YOLO_MODEL = YOLO(yolo_path)
        print("[CV ENGINE] Loaded HuggingFace Fine-Tuned Football YOLOv8 Model.", flush=True)
    return _YOLO_MODEL


def get_pnlcalib_models(frame_w, frame_h):
    """Load PnLCalib HRNet keypoint + line models (same as 03_run_calibration.py)."""
    global _PNLCALIB_MODELS
    if _PNLCALIB_MODELS is None:
        import torch
        import yaml
        import torchvision.transforms as T

        # Import PnLCalib modules
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib
        import inference as inf_mod

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inf_mod.device = device
        inf_mod.transform2 = T.Resize((540, 960))

        cfg_path = os.path.join(PNLCALIB_DIR, "config", "hrnetv2_w48.yaml")
        cfg_l_path = os.path.join(PNLCALIB_DIR, "config", "hrnetv2_w48_l.yaml")

        cfg = yaml.safe_load(open(cfg_path, 'r'))
        cfg_l = yaml.safe_load(open(cfg_l_path, 'r'))

        weights_kp = os.path.join(PNLCALIB_DIR, "weights", "SV_kp")
        weights_line = os.path.join(PNLCALIB_DIR, "weights", "SV_lines")

        print(f"[CV ENGINE] Loading PnLCalib HRNet keypoint model on {device}...", flush=True)
        loaded_state = torch.load(weights_kp, map_location=device)
        model_kp = get_cls_net(cfg)
        model_kp.load_state_dict(loaded_state)
        model_kp.to(device).eval()

        print(f"[CV ENGINE] Loading PnLCalib HRNet line model on {device}...", flush=True)
        loaded_state_l = torch.load(weights_line, map_location=device)
        model_line = get_cls_net_l(cfg_l)
        model_line.load_state_dict(loaded_state_l)
        model_line.to(device).eval()

        cam_calib = FramebyFrameCalib(iwidth=frame_w, iheight=frame_h, denormalize=True)

        _PNLCALIB_MODELS = (model_kp, model_line, cam_calib)
        print("[CV ENGINE] PnLCalib HRNet models loaded successfully.", flush=True)

    return _PNLCALIB_MODELS


# ---------------------------------------------------------------------------
# Stage A: Bootstrap Team Color Discovery (from bootstrap_team_colors.py)
# ---------------------------------------------------------------------------
def bootstrap_team_colors(video_path, num_sample_frames=30):
    """
    Discovers 2 team color centroids via circular-hue K-Means (k=2) on
    torso crops from sampled frames. Exact logic from bootstrap_team_colors.py.
    """
    print(f"[CV ENGINE] Stage A: Bootstrap Team Color Discovery ({num_sample_frames} frames)...", flush=True)

    yolo_model = get_yolo_model()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_sample_frames).astype(int)

    feature_vectors = []
    raw_hsv_samples = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        results = yolo_model.predict(frame, device=0, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_name = yolo_model.names[int(box.cls)].lower()
                if cls_name == "player":
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    crop = get_torso_crop(frame, x1, y1, x2, y2)
                    hsv = extract_dominant_hsv(crop)
                    hue, sat = hsv["hue"], hsv["saturation"]

                    theta = (hue / 180.0) * (2.0 * math.pi)
                    cos_val = math.cos(theta)
                    sin_val = math.sin(theta)
                    sat_norm = sat / 255.0

                    raw_hsv_samples.append({"hue": hue, "saturation": sat})
                    feature_vectors.append([cos_val, sin_val, sat_norm])

    cap.release()
    print(f"[CV ENGINE]   Collected {len(feature_vectors)} player torso HSV samples.", flush=True)

    if len(feature_vectors) < 10:
        print("[CV ENGINE]   WARNING: Too few samples, using fallback centroids.", flush=True)
        return [
            {"cluster_id": 0, "hue": 5.0, "saturation": 200.0},
            {"cluster_id": 1, "hue": 105.0, "saturation": 180.0}
        ]

    X = np.array(feature_vectors, dtype=np.float64)
    best_centers, best_labels = _kmeans_k2(X)

    centroids = []
    for c_idx, center in enumerate(best_centers):
        c_cos, c_sin, c_sat_norm = center[0], center[1], center[2]
        theta = math.atan2(c_sin, c_cos)
        if theta < 0:
            theta += 2.0 * math.pi
        hue_deg = (theta / (2.0 * math.pi)) * 180.0
        sat_val = c_sat_norm * 255.0
        centroids.append({
            "cluster_id": c_idx,
            "hue": round(float(hue_deg), 2),
            "saturation": round(float(sat_val), 2)
        })

    for c in centroids:
        print(f"[CV ENGINE]   Cluster {c['cluster_id']}: Hue={c['hue']}deg, Sat={c['saturation']}", flush=True)

    return centroids


def _kmeans_k2(X, max_iter=100, n_init=10, seed=42):
    """Pure NumPy K-Means k=2 (from bootstrap_team_colors.py)."""
    np.random.seed(seed)
    best_inertia = float("inf")
    best_centers = None
    best_labels = None

    for init in range(n_init):
        indices = np.random.choice(len(X), size=2, replace=False)
        centers = X[indices].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(2):
                cluster_pts = X[labels == k]
                if len(cluster_pts) > 0:
                    new_centers[k] = cluster_pts.mean(axis=0)
                else:
                    new_centers[k] = X[np.random.choice(len(X))]

            if np.allclose(centers, new_centers, atol=1e-5):
                break
            centers = new_centers

        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        inertia = np.sum(np.min(dists, axis=1) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return best_centers, best_labels


def select_target_cluster(centroids, kit_color_input):
    """
    Determines which discovered cluster matches the user's kit color selection.
    """
    KIT_HSV_TARGETS = {
        "sky_blue": 105.0, "red": 5.0, "white": 0.0,
        "yellow": 28.0, "emerald": 65.0, "navy": 115.0,
    }

    key = str(kit_color_input).lower().replace("-", "_").replace(" ", "_")
    if key in KIT_HSV_TARGETS:
        target_hue = KIT_HSV_TARGETS[key]
    elif str(kit_color_input).startswith("#"):
        hex_code = kit_color_input.lstrip("#")
        if len(hex_code) == 6:
            r = int(hex_code[0:2], 16)
            g = int(hex_code[2:4], 16)
            b = int(hex_code[4:6], 16)
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            target_hue = float(hsv[0])
        else:
            target_hue = 105.0
    else:
        target_hue = 105.0

    c0_dist = circular_hue_distance(centroids[0]["hue"], target_hue)
    c1_dist = circular_hue_distance(centroids[1]["hue"], target_hue)

    if c0_dist <= c1_dist:
        target_idx = 0
    else:
        target_idx = 1

    opp_idx = 1 - target_idx
    print(f"[CV ENGINE]   Selected Cluster {target_idx} as TARGET (hue_dist={min(c0_dist, c1_dist):.1f})", flush=True)
    return centroids[target_idx], centroids[opp_idx]


# ---------------------------------------------------------------------------
# Stage B: PnLCalib Homography (from 02_run_phase_B_benchmark.py)
# ---------------------------------------------------------------------------
def compute_homography_from_params(params_dict):
    """
    Computes homography H_inv and pitch polygon from PnLCalib camera params.
    Exact logic from 02_run_phase_B_benchmark.py compute_homography_and_projection_from_json().
    """
    from inference import projection_from_cam_params

    cam_params = params_dict["cam_params"]
    fx = cam_params["x_focal_length"]
    fy = cam_params["y_focal_length"]
    cx, cy = cam_params["principal_point"]
    pos = np.array(cam_params["position_meters"])
    R = np.array(cam_params["rotation_matrix"])

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    t = -R @ pos
    H_3d = K @ np.column_stack((R[:, 0], R[:, 1], t))

    try:
        H_inv = np.linalg.inv(H_3d)
    except np.linalg.LinAlgError:
        return None, None

    return H_inv, H_3d


def pixel_to_pitch(H_inv, px, py):
    """Convert pixel [u, v] to pitch [x, y] meters via homography inverse."""
    p_img = np.array([float(px), float(py), 1.0], dtype=np.float64)
    p_pitch = H_inv @ p_img
    if abs(p_pitch[2]) < 1e-6:
        return None
    return float(p_pitch[0] / p_pitch[2]), float(p_pitch[1] / p_pitch[2])


def get_pitch_polygon_2d(H_3d):
    """
    Projects pitch corners to image space for polygon test.
    Exact logic from 02_run_phase_B_benchmark.py get_2d_pitch_polygon().
    """
    pitch_corners_3d = [
        [-52.5, -34.0, 1.0],
        [52.5, -34.0, 1.0],
        [52.5, 34.0, 1.0],
        [-52.5, 34.0, 1.0]
    ]
    pts_2d = []
    for corner in pitch_corners_3d:
        p_img = H_3d @ np.array(corner, dtype=np.float64)
        if abs(p_img[2]) > 1e-6:
            pts_2d.append([p_img[0] / p_img[2], p_img[1] / p_img[2]])
    if len(pts_2d) == 4:
        return np.array(pts_2d, dtype=np.int32)
    return None


# ---------------------------------------------------------------------------
# Stage D: IoU Temporal Smoothing
# ---------------------------------------------------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


class MultiFrameIoUTracker:
    def __init__(self, iou_thresh=0.35, history_len=5):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.history_len = history_len

    def update(self, current_detections):
        updated_tracks = {}
        matched_det_indices = set()

        for tid, track in self.tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for idx, det in enumerate(current_detections):
                if idx in matched_det_indices:
                    continue
                iou = compute_iou(track["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_iou >= self.iou_thresh and best_det_idx >= 0:
                matched_det_indices.add(best_det_idx)
                det = current_detections[best_det_idx]

                pitch_hist = track["pitch_hist"] + [det["pitch"]]
                if len(pitch_hist) > self.history_len:
                    pitch_hist = pitch_hist[-self.history_len:]

                avg_pitch = np.mean(pitch_hist, axis=0).tolist()

                updated_tracks[tid] = {
                    "box": det["box"],
                    "pitch": [round(avg_pitch[0], 2), round(avg_pitch[1], 2)],
                    "pitch_hist": pitch_hist,
                }

        for idx, det in enumerate(current_detections):
            if idx not in matched_det_indices:
                tid = self.next_id
                self.next_id += 1
                updated_tracks[tid] = {
                    "box": det["box"],
                    "pitch": det["pitch"],
                    "pitch_hist": [det["pitch"]],
                }

        self.tracks = updated_tracks
        return self.tracks


# ---------------------------------------------------------------------------
# HSV Distance (from team_classifier.py)
# ---------------------------------------------------------------------------
def compute_hsv_distance(hue, sat, ref_h, ref_s):
    hue_dist = circular_hue_distance(hue, ref_h)
    sat_dist = abs(sat - ref_s)
    return 1.0 * hue_dist + 0.3 * sat_dist


# ---------------------------------------------------------------------------
# Main Entry Point: process_video_clip
# ---------------------------------------------------------------------------
def process_video_clip(video_path, kit_color_input, target_fps=25.0):
    """
    Full production pipeline:
    1. Bootstrap team colors from video
    2. Load PnLCalib HRNet models
    3. Per-frame: PnLCalib calibration → YOLO detection → HSV classification → IoU tracking
    4. Return position dataset for dashboard
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or target_fps)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(1.0, fps)
    cap.release()

    print(f"[CV ENGINE] Video: {video_path}", flush=True)
    print(f"   Frames: {total_frames} | FPS: {fps:.2f} | Resolution: {frame_w}x{frame_h}", flush=True)
    print(f"   Duration: {duration_sec:.1f}s ({duration_sec/60:.1f}m)", flush=True)

    # ── Stage A: Bootstrap Team Color Discovery ──
    centroids = bootstrap_team_colors(video_path, num_sample_frames=30)
    target_centroid, opponent_centroid = select_target_cluster(centroids, kit_color_input)

    # ── Stage B: Load PnLCalib HRNet Models ──
    model_kp, model_line, cam_calib = get_pnlcalib_models(frame_w, frame_h)
    from inference import inference as pnl_inference

    # ── Stage C+D: Per-frame detection + calibration ──
    yolo_model = get_yolo_model()
    tracker = MultiFrameIoUTracker(iou_thresh=0.35, history_len=5)

    cap = cv2.VideoCapture(video_path)
    stride = max(1, int(fps / target_fps))

    extracted_frames = []
    frame_idx = 0
    last_valid_H_inv = None
    last_valid_H_3d = None
    calib_solved = 0
    calib_failed = 0

    print(f"[CV ENGINE] Stage B+C: Per-frame PnLCalib + YOLO + HSV Processing (stride={stride})...", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            # ── PnLCalib Inference ──
            params_dict = pnl_inference(
                cam_calib, frame, model_kp, model_line,
                kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=False
            )

            if params_dict is not None:
                H_inv, H_3d = compute_homography_from_params(params_dict)
                if H_inv is not None:
                    last_valid_H_inv = H_inv
                    last_valid_H_3d = H_3d
                    calib_solved += 1
                else:
                    calib_failed += 1
            else:
                calib_failed += 1

            # Use last valid calibration if current failed
            H_inv = last_valid_H_inv
            H_3d = last_valid_H_3d

            if H_inv is None:
                frame_idx += 1
                continue

            # Pitch polygon for filtering
            pitch_poly = get_pitch_polygon_2d(H_3d)

            # ── YOLO Detection ──
            results = yolo_model.predict(frame, device=0, verbose=False)
            current_dets = []

            for r in results:
                for box_obj in r.boxes:
                    conf = float(box_obj.conf)
                    cls_name = yolo_model.names[int(box_obj.cls)].lower()

                    if conf < 0.50:
                        continue

                    x1, y1, x2, y2 = box_obj.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1

                    # Aspect ratio filter (same as 02_run_phase_B)
                    aspect_ratio = w / float(h + 1e-6)
                    if not (0.25 <= aspect_ratio <= 0.85):
                        continue

                    if cls_name == "referee":
                        continue

                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2

                    # Pitch polygon test
                    if pitch_poly is not None:
                        is_in_poly = cv2.pointPolygonTest(
                            pitch_poly, (float(foot_x), float(foot_y)), False
                        )
                        if is_in_poly < 0:
                            continue

                    # Pixel → Pitch projection
                    pitch_pt = pixel_to_pitch(H_inv, foot_x, foot_y)
                    if pitch_pt is None:
                        continue

                    pitch_X, pitch_Y = pitch_pt
                    if not (-52.5 <= pitch_X <= 52.5 and -34.0 <= pitch_Y <= 34.0):
                        continue

                    # Torso HSV Classification
                    crop = get_torso_crop(frame, x1, y1, x2, y2)
                    hsv = extract_dominant_hsv(crop)
                    h_val, s_val = hsv["hue"], hsv["saturation"]

                    d_target = compute_hsv_distance(
                        h_val, s_val,
                        target_centroid["hue"], target_centroid["saturation"]
                    )
                    d_opp = compute_hsv_distance(
                        h_val, s_val,
                        opponent_centroid["hue"], opponent_centroid["saturation"]
                    )

                    if d_target <= d_opp:
                        current_dets.append({
                            "box": [x1, y1, x2, y2],
                            "pitch": [round(pitch_X, 2), round(pitch_Y, 2)],
                        })

            # Top-10 selection (same as Phase B)
            current_dets = current_dets[:10]

            # ── IoU Tracking ──
            active_tracks = tracker.update(current_dets)

            target_positions = []
            for tid, trk in active_tracks.items():
                target_positions.append(trk["pitch"])

            if len(target_positions) > 0:
                extracted_frames.append({
                    "frame_idx": len(extracted_frames),
                    "timestamp_sec": round(len(extracted_frames) / target_fps, 2),
                    "target_positions": target_positions
                })

            if len(extracted_frames) % 50 == 0 and len(extracted_frames) > 0:
                pct = round(frame_idx / max(1, total_frames) * 100, 1)
                print(
                    f"   [{pct}%] {len(extracted_frames)} frames extracted "
                    f"(calib solved: {calib_solved}, failed: {calib_failed})",
                    flush=True
                )

        frame_idx += 1

    cap.release()

    calib_rate = round(calib_solved / max(1, calib_solved + calib_failed) * 100, 1)
    print(f"[CV ENGINE] COMPLETE: {len(extracted_frames)} frames extracted.", flush=True)
    print(f"   PnLCalib solve rate: {calib_rate}% ({calib_solved}/{calib_solved + calib_failed})", flush=True)

    return extracted_frames, fps, total_frames, duration_sec
