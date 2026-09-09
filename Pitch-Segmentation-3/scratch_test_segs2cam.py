"""
Test segs2cam on input_video frame
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOCCERSEGCAL_DIR = os.path.join(SCRIPT_DIR, "soccersegcal")
if SOCCERSEGCAL_DIR not in sys.path:
    sys.path.insert(0, SOCCERSEGCAL_DIR)

import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T

from soccersegcal.pose import segs2cam
from sncalib.baseline_cameras import Camera

class LitSoccerFieldSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(num_classes=6)

    def forward(self, x):
        return self.model(x)['out']

def main():
    ckpt_path = os.path.join(SCRIPT_DIR, "weights", "snapshot.ckpt")
    video_path = r"C:\CLG\PP1\EXP\Pitch-Segmentation\data\input_video.mp4"

    model = LitSoccerFieldSegmentation.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Cannot read frame")
        return

    frame_resized = cv2.resize(frame, (960, 540))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        segs = torch.sigmoid(model(tensor_img))[0].cpu()

    world_scale = 100
    print("[INFO] Running Spiideo segs2cam 3D camera solver...")
    ptz_model = segs2cam(segs, world_scale, None, show=False)

    if ptz_model is not None:
        ptz_model = ptz_model.cpu()
        smalles_image_side = min(segs.shape[2], segs.shape[1])
        f = smalles_image_side / 2 / ptz_model.camera_focal.item()
        cam = Camera(segs.shape[2], segs.shape[1])
        cam.from_json_parameters({
            'position_meters': ptz_model.camera_position.detach().numpy() * world_scale,
            'principal_point': cam.principal_point,
            'x_focal_length': f,
            'y_focal_length': f,
            'pan_degrees': np.rad2deg(ptz_model.camera_pan.item()),
            'tilt_degrees': np.rad2deg(ptz_model.camera_tilt.item()),
            'roll_degrees': np.rad2deg(ptz_model.camera_roll.item()),
            'radial_distortion': ptz_model.radial_distortion.detach().numpy() if hasattr(ptz_model, 'radial_distortion') else np.zeros(6),
            'tangential_distortion': ptz_model.tangential_disto.detach().numpy() if hasattr(ptz_model, 'tangential_disto') else np.zeros(2),
            'thin_prism_distortion': ptz_model.thin_prism_disto.detach().numpy() if hasattr(ptz_model, 'thin_prism_disto') else np.zeros(4),
        })
        print(f"[SUCCESS] Camera Parameters Solved! Pan={np.rad2deg(ptz_model.camera_pan.item()):.2f}°, Tilt={np.rad2deg(ptz_model.camera_tilt.item()):.2f}°")
    else:
        print("[FAIL] segs2cam returned None")

if __name__ == "__main__":
    main()
