"""
Diagnostic script to inspect Spiideo snapshot.ckpt predictions
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
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

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
    tensor_img = transform(frame_rgb).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        logits = model(tensor_img)[0]
        preds = torch.sigmoid(logits).cpu().numpy()

    print(f"[DIAGNOSTIC] Predictions Shape: {preds.shape}")
    for ch in range(preds.shape[0]):
        mask = preds[ch]
        pixels_above_half = np.sum(mask > 0.5)
        print(f" -> Channel {ch}: Min={mask.min():.4f}, Max={mask.max():.4f}, Mean={mask.mean():.4f}, Count(>0.5)={pixels_above_half}")

if __name__ == "__main__":
    main()
