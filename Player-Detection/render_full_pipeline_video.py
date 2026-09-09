"""
Full Pipeline Video Renderer
Combines, per frame: PnLCalib calibration -> YOLO detection -> pitch-boundary +
logo filtering (detection_filters.py) -> draws two output videos:
  - "clean" video: only kept (on-pitch, non-logo) detections + pitch boundary
  - "debug" video: same, plus rejected detections shown (gray=off-pitch, magenta=logo)

No calibration or detection math is reimplemented here — reuses PnLCalib's own
inference() function and the existing YOLO model, same as prior tested scripts.
"""
import os
import sys
import cv2
import numpy as np
import torch
import yaml
import argparse
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

PNLCALIB_DIR = r"C:\CLG\PP1\EXP\PnLCalib"
SN_CALIB_DIR = r"C:\CLG\PP1\EXP\Pitch-Calibration-Clean\sn-calibration"
sys.path.insert(0, PNLCALIB_DIR)
sys.path.insert(0, SN_CALIB_DIR)

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
import inference as pnl_inference_module
from src.camera import Camera
from src.soccerpitch import SoccerPitch

sys.path.insert(0, r"C:\CLG\PP1\EXP\Player-Detection")
from detection_filters import filter_detections

WEIGHTS_KP = os.path.join(PNLCALIB_DIR, "weights", "SV_kp")
WEIGHTS_LINE = os.path.join(PNLCALIB_DIR, "weights", "SV_lines")
DEVICE = "cuda:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_clean", type=str, required=True)
    parser.add_argument("--output_debug", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=None,
                         help="Optional cap for a quick test run before committing to the full clip")
    args = parser.parse_args()

    # --- Load calibration models ---
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

    pnl_inference_module.transform2 = T.Resize((540, 960))
    pnl_inference_module.device = DEVICE

    # --- Load YOLO detector ---
    yolo_path = hf_hub_download(repo_id="uisikdag/yolo-v8-football-players-detection", filename="best.pt")
    yolo_model = YOLO(yolo_path)

    # --- Load pitch model ---
    pitch = SoccerPitch()

    # --- Video setup ---
    cap = cv2.VideoCapture(args.input_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    print(f"[INFO] {args.input_path}: {frame_w}x{frame_h} @ {fps:.2f}fps, processing {total_frames} frames")

    os.makedirs(os.path.dirname(args.output_clean), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_debug), exist_ok=True)
    writer_clean = cv2.VideoWriter(args.output_clean, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    writer_debug = cv2.VideoWriter(args.output_debug, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    cam_calib = FramebyFrameCalib(iwidth=frame_w, iheight=frame_h, denormalize=True)

    frame_idx = 0
    calib_success = 0
    calib_fail = 0
    total_kept = 0
    total_rejected_offpitch = 0
    total_rejected_logo = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_frames and frame_idx >= args.max_frames):
            break
        frame_idx += 1

        # --- Calibration ---
        final_params_dict = pnl_inference_module.inference(
            cam_calib, frame, kp_model, line_model, 0.3434, 0.7867, False
        )

        camera_obj = None
        if final_params_dict is not None:
            calib_success += 1
            cam_params = final_params_dict["cam_params"]
            calib_w = round(cam_params["principal_point"][0] * 2)
            calib_h = round(cam_params["principal_point"][1] * 2)
            camera_obj = Camera(calib_w, calib_h)
            camera_obj.from_json_parameters(cam_params)
            if (frame_w, frame_h) != (calib_w, calib_h):
                camera_obj.scale_resolution(frame_w / calib_w)
        else:
            calib_fail += 1

        # --- Detection ---
        results = yolo_model.predict(frame, device=0, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((cls_name, conf, x1, y1, x2, y2))

        # --- Filtering ---
        kept, rejected = filter_detections(detections, camera_obj, pitch)
        total_kept += len(kept)
        total_rejected_offpitch += sum(1 for d in rejected if d[6] == "off_pitch")
        total_rejected_logo += sum(1 for d in rejected if d[6] == "logo_overlay")

        # --- Pitch boundary polygon (for drawing) ---
        polygon_np = None
        if camera_obj is not None:
            corner_keys = ["TL_PITCH_CORNER", "TR_PITCH_CORNER", "BR_PITCH_CORNER", "BL_PITCH_CORNER"]
            pts = []
            for key in corner_keys:
                pt2d = camera_obj.project_point(pitch.point_dict[key])
                if not np.isnan(pt2d).any():
                    pts.append((float(pt2d[0]), float(pt2d[1])))
            if len(pts) == 4:
                polygon_np = np.array(pts, dtype=np.float32)

        # --- Draw CLEAN version ---
        clean_frame = frame.copy()
        if polygon_np is not None:
            cv2.polylines(clean_frame, [polygon_np.astype(int)], True, (0, 255, 255), 2)
        for cls_name, conf, x1, y1, x2, y2, reason in kept:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 255, 0) if cls_name in ("player", "goalkeeper") else \
                    (0, 0, 255) if cls_name == "ball" else (255, 0, 0)
            cv2.rectangle(clean_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(clean_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        writer_clean.write(clean_frame)

        # --- Draw DEBUG version (clean + rejected boxes) ---
        debug_frame = clean_frame.copy()
        for cls_name, conf, x1, y1, x2, y2, reason in rejected:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (255, 0, 255) if reason == "logo_overlay" else (128, 128, 128)
            label = "LOGO REJECTED" if reason == "logo_overlay" else "REJECTED"
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        writer_debug.write(debug_frame)

        if frame_idx % 100 == 0:
            print(f"[PROGRESS] {frame_idx}/{total_frames} | calib_ok={calib_success} calib_fail={calib_fail} "
                  f"| kept={total_kept} rej_offpitch={total_rejected_offpitch} rej_logo={total_rejected_logo}")

    cap.release()
    writer_clean.release()
    writer_debug.release()

    print(f"\n[SUCCESS] Done. Frames processed: {frame_idx}")
    print(f"[SUMMARY] Calibration: {calib_success} ok / {calib_fail} failed")
    print(f"[SUMMARY] Detections: {total_kept} kept | {total_rejected_offpitch} off-pitch rejected | {total_rejected_logo} logo rejected")
    print(f"[SUCCESS] Clean video: {args.output_clean}")
    print(f"[SUCCESS] Debug video: {args.output_debug}")


if __name__ == "__main__":
    main()
