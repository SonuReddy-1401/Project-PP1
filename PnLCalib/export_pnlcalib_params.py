"""
PnLCalib Camera Parameter Exporter
Reuses PnLCalib's own inference() function exactly as inference.py does, but
additionally dumps final_params_dict to a JSON file per frame, so we can later
project pitch boundaries / player foot-points using these parameters, the same
way we did with sn-calibration's camera_XXXXX.json files.

No homography/camera math is reimplemented here — this only adds an export step
around PnLCalib's existing, tested pipeline.
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import yaml
import torchvision.transforms as T
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib

# Import inference module and bind required module-level variables
import inference as inf_mod
from inference import inference

inf_mod.transform2 = T.Resize((540, 960))


def make_json_safe(obj):
    """Recursively convert numpy types to plain Python types for JSON export."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_kp", type=str, required=True)
    parser.add_argument("--weights_line", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Folder to write per-frame camera_XXXXX.json files")
    parser.add_argument("--kp_threshold", type=float, default=0.3434)
    parser.add_argument("--line_threshold", type=float, default=0.7867)
    parser.add_argument("--pnl_refine", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    inf_mod.device = device

    cfg = yaml.safe_load(open(os.path.join(SCRIPT_DIR, "config", "hrnetv2_w48.yaml"), 'r'))
    cfg_l = yaml.safe_load(open(os.path.join(SCRIPT_DIR, "config", "hrnetv2_w48_l.yaml"), 'r'))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    cap = cv2.VideoCapture(args.input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

    frame_idx = 0
    solved_count = 0
    failed_count = 0

    pbar = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        final_params_dict = inference(cam, frame, model, model_l,
                                       args.kp_threshold, args.line_threshold, args.pnl_refine)

        if final_params_dict is not None:
            solved_count += 1
            out_path = os.path.join(args.output_dir, f"frame_{frame_idx:05d}.json")
            with open(out_path, "w") as f:
                json.dump(make_json_safe(final_params_dict), f, indent=2)
        else:
            failed_count += 1

        pbar.update(1)
        if frame_idx % 100 == 0:
            print(f"[PROGRESS] {frame_idx}/{total_frames} | Solved: {solved_count} | Failed: {failed_count}")

    cap.release()
    print(f"\n[SUCCESS] Done. Solved: {solved_count} | Failed: {failed_count}")
    print(f"[SUCCESS] JSON files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
