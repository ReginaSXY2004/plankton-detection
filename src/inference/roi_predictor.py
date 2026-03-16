import os
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from models.roi_classifier import SimpleCNN


class ROIPredictor:
    def __init__(self, weight_path=None, device=None):
        base_dir = Path(__file__).resolve().parent.parent.parent

        if weight_path is None:
            weight_path = base_dir / "weights" / "best_roi_classifier.pth"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.class_names = ["junk", "target"]

        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def predict(self, roi_bgr):
        # OpenCV 是 BGR，训练时 ImageFolder/PIL 实际按 RGB 读图
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        x = self.transform(roi_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()

        pred_label = self.class_names[pred_idx]
        return pred_label, conf