import os
import urllib.request
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np

# ----------------------------
# Auto-download model weights
# ----------------------------
MODEL_DIR = "models"
TORCH_WEIGHTS = os.path.join(MODEL_DIR, "xception_ffpp.pth")
MODEL_URL = "https://github.com/ondyari/FaceForensics/raw/master/models/xception.pth"  # FaceForensics++ Xception

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(TORCH_WEIGHTS):
    print(f"[DeepfakeModel] Downloading pretrained weights to {TORCH_WEIGHTS}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, TORCH_WEIGHTS)
        print("[DeepfakeModel] Download complete.")
    except Exception as e:
        print("[DeepfakeModel] ERROR downloading weights:", e)

# ----------------------------
# REAL Xception model definition
# ----------------------------
class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, stride=1, grow_first=True):
        super().__init__()
        layers = []
        filters = in_filters
        if grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_filters, out_filters, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(filters, filters, 3, padding=1))
            layers.append(nn.BatchNorm2d(filters))

        if not grow_first:
            layers = [
                nn.ReLU(inplace=True),
                nn.Conv2d(in_filters, out_filters, 3, padding=1),
                nn.BatchNorm2d(out_filters)
            ] + layers

        if stride != 1:
            self.residual = nn.Conv2d(in_filters, out_filters, 1, stride=stride)
        else:
            self.residual = None

        layers.append(nn.MaxPool2d(3, stride=stride, padding=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.residual is not None:
            x = self.residual(x)
        return out + x

class XceptionDeepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            XceptionBlock(64, 128, 2, stride=2),
            XceptionBlock(128, 256, 2, stride=2),
            XceptionBlock(256, 728, 2, stride=2),
        )

        self.exit = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(728, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        return x

# ----------------------------
# Torch Deepfake wrapper
# ----------------------------
class TorchDeepfakeWrapper:
    def __init__(self, weights_path=TORCH_WEIGHTS, input_size=(160,160), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        self.input_size = input_size

        self.model = XceptionDeepfake().to(self.device)

        if os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(ckpt)
                print("[DeepfakeModel] Loaded pretrained Xception weights.")
            except Exception as e:
                print("[DeepfakeModel] Failed to load weights:", e)
        else:
            print("[DeepfakeModel] WARNING: Weights file not found, using random init.")

        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def infer(self, frames):
        """frames: list of BGR images. Returns list of probabilities of being fake."""
        with torch.no_grad():
            batch = []
            for f in frames:
                try:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    batch.append(self.transform(rgb))
                except Exception:
                    batch.append(torch.zeros(3, self.input_size[0], self.input_size[1]))

            x = torch.stack(batch).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
            return probs
