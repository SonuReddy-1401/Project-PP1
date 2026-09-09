"""
10-Frame Robustness Test — Detection + PnLCalib Calibration + Integrated Pitch Filtering
Randomly samples 10 frames from broadcast_clip_1.mp4 (the harder 2.5min clip),
runs YOLO detection + PnLCalib calibration + integrated detection_filters (logo exclusion
+ geometric pitch-boundary filtering), and saves annotated images + summary JSON.
"""
import os
import sys
import json
import random
import cv2
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

PNLCALIB_DIR = r"C:\CLG\PP1\EXP\PnLCalib"
SN_CALIB_DIR = r"C:\CLG\PP1\EXP\Pitch-Calibration-Clean\sn-calibration"
DETECTION_DIR = r"C:\CLG\PP1\EXP\Player-Detection"

sys.path.insert(0, PNLCALIB_DIR)
sys.path.insert(0, SN_CALIB_DIR)
sys.path.insert(0, DETECTION_DIR)

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
import inference as pnl_inference_module
from src.camera import Camera
from src.soccerpitch import SoccerPitch
from detection_filters import filter_detections, build_pitch_polygon

VIDEO_PATH = os.path.join(PNLCALIB_DIR, "data", "broadcast_clip_1.mp4")
WEIGHTS_KP = os.path.join(PNLCALIB_DIR, "weights", "SV_kp")
WEIGHTS_LINE = os.path.join(PNLCALIB_DIR, "weights", "SV_lines")
OUTPUT_DIR = r"C:\CLG\PP1\EXP\Player-Detection\robustness_test_10frames"
DEVICE = "cuda:0"
NUM_TEST_FRAMES = 10
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    # --- Load calibration models (PnLCalib) ---
    cfg = yaml.safe_load(open(os.path.join(PNLCALIB_DIR, "config/hrnetv2_w48.yaml"), 'r'))
    cfg_l = yaml.safe_load(open(os.path.join(PNLCALIB_DIR, "config/hrnetv2_w48_l.yaml"), 'r'))

    kp_model = get_cls_net(cfg)
    kp_model.load_state_dict(torch.load(WEIGHTS_KP, map_location=DEVICE))
    kp_model.to(DEVICE)
    kp_model.eval()

    line_model = get_cls_net_l(cfg_l)
    line_model.load_state_dict(torch.load(WEIGHTS_LINE, map_location=DEVICE))
    line_model.to(DEVICE)
    line_model.eval()

    # Bind globals inference() expects
    import torchvision.transforms as T
    pnl_inference_module.transform2 = T.Resize((540, 960))
    pnl_inference_module.device = DEVICE

    # --- Load YOLO detector ---
    yolo_path = hf_hub_download(repo_id="uisikdag/yolo-v8-football-players-detection", filename="best.pt")
    yolo_model = YOLO(yolo_path)

    # --- Load pitch model ---
    pitch = SoccerPitch()

    # --- Pick random frame indices ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video reports {total_frames} frames at {frame_w}x{frame_h}")

    random.seed(RANDOM_SEED)
    safe_max = max(1, total_frames - 200)
    frame_indices = sorted(random.sample(range(safe_max), NUM_TEST_FRAMES))
    print(f"[INFO] Testing frame indices: {frame_indices}")

    cam_calib = FramebyFrameCalib(iwidth=frame_w, iheight=frame_h, denormalize=True)

    summary = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Could not read frame {idx}, skipping")
            continue

        # --- Calibration ---
        final_params_dict = pnl_inference_module.inference(
            cam_calib, frame, kp_model, line_model, 0.3434, 0.7867, False
        )

        if final_params_dict is None:
            print(f"[FRAME {idx}] Calibration FAILED — skipping pitch filtering, saving raw detections only")
            calib_ok = False
            camera_obj = None
            rep_err = None
        else:
            calib_ok = True
            cam_params = final_params_dict["cam_params"]
            rep_err = final_params_dict.get("rep_err", None)
            calib_w = round(cam_params["principal_point"][0] * 2)
            calib_h = round(cam_params["principal_point"][1] * 2)
            camera_obj = Camera(calib_w, calib_h)
            camera_obj.from_json_parameters(cam_params)
            if (frame_w, frame_h) != (calib_w, calib_h):
                camera_obj.scale_resolution(frame_w / calib_w)

        # --- Detection ---
        results = yolo_model.predict(frame, device=0, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((cls_name, conf, x1, y1, x2, y2))

        # --- Integrated Filtering ---
        kept, rejected = filter_detections(detections, camera_obj, pitch)

        polygon_np = build_pitch_polygon(camera_obj, pitch) if camera_obj is not None else None

        # --- Draw and save ---
        annotated = frame.copy()
        if polygon_np is not None:
            cv2.polylines(annotated, [polygon_np.astype(int)], isClosed=True, color=(0, 255, 255), thickness=3)

        for cls_name, conf, x1, y1, x2, y2, reason in kept:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 255, 0) if cls_name == "player" else (0, 0, 255) if cls_name == "ball" else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        logo_rejected_count = 0
        off_pitch_rejected_count = 0

        for cls_name, conf, x1, y1, x2, y2, reason in rejected:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if reason == "logo_overlay":
                logo_rejected_count += 1
                box_color = (255, 0, 255)  # Magenta for logo overlay rejection
                label = "LOGO REJECTED"
            else:
                off_pitch_rejected_count += 1
                box_color = (128, 128, 128)  # Gray for off-pitch boundary rejection
                label = "REJECTED"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:05d}_annotated.jpg")
        cv2.imwrite(out_path, annotated)

        summary.append({
            "frame_idx": idx,
            "calibration_ok": calib_ok,
            "rep_err": rep_err if calib_ok else None,
            "total_detections": len(detections),
            "kept": len(kept),
            "rejected_total": len(rejected),
            "rejected_logo_overlay": logo_rejected_count,
            "rejected_off_pitch": off_pitch_rejected_count,
        })
        print(f"[FRAME {idx}] calib_ok={calib_ok} rep_err={summary[-1]['rep_err']} "
              f"kept={len(kept)} rejected_total={len(rejected)} (logo={logo_rejected_count}, off_pitch={off_pitch_rejected_count}) -> {out_path}")

    cap.release()

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(make_json_safe(summary), f, indent=2)

    print(f"\n[SUCCESS] Done. Integrated filter summary saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
