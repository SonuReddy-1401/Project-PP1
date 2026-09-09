"""
Extract 3D-2D keypoints from Spiideo 6-class area segmentations and solve camera parameters.
"""

import os
import sys
import cv2
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOCCERSEGCAL_DIR = os.path.join(SCRIPT_DIR, "soccersegcal")
if SOCCERSEGCAL_DIR not in sys.path:
    sys.path.insert(0, SOCCERSEGCAL_DIR)

import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T

from sncalib.soccerpitch import SoccerPitch
from sncalib.baseline_cameras import Camera


class LitSoccerFieldSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(num_classes=6)

    def forward(self, x):
        return self.model(x)['out']


def get_corners_from_mask(mask: np.ndarray):
    """Find convex hull vertices / corners of a binary region mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None

    hull = cv2.convexHull(c)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2)


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

    width, height = 960, 540
    frame_resized = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(tensor_img))[0].cpu().numpy()

    # Channel 0: FullField, Channel 1: CircleCentral, Channel 2: BigRect, Channel 3: CircleSide, Channel 4: SmallRect, Channel 5: Goal
    pitch = SoccerPitch()
    point_matches = []

    # 1. BigRect (Penalty area)
    big_rect_mask = (preds[2] > 0.5).astype(np.uint8)
    big_rect_pts = get_corners_from_mask(big_rect_mask)
    if big_rect_pts is not None:
        print(f"[FOUND] BigRect (Penalty Box) Corners: {len(big_rect_pts)} points")

    # 2. CircleCentral
    circle_mask = (preds[1] > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            print(f"[FOUND] Center Circle Center: ({cx:.1f}, {cy:.1f})")

if __name__ == "__main__":
    main()
