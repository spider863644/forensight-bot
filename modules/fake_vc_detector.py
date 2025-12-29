#!/usr/bin/env python3
"""
Cross-Platform Real-Time Fake Video Call Detector (Modular)
Windows / macOS / Linux

Audio: soundcard loopback
Video: screen capture via mss + face detection + deepfake model
"""

import os
import time
import threading
import numpy as np
import sounddevice as sd
import cv2
import mss
from scipy.signal import find_peaks
from collections import deque
import urllib.request

# optional soundcard import (only used elsewhere if needed)
try:
    import soundcard as sc
except Exception:
    sc = None

# ----------------------------
# Auto-download model weights (optional, can be disabled)
# ----------------------------
MODEL_DIR = "models"
TORCH_WEIGHTS = os.path.join(MODEL_DIR, "ffpp_c40.pth")
GDRIVE_FILE_ID = "1jyvW3GCmYqS6gItcw50ot2GI2LrZACwc"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(TORCH_WEIGHTS):
    try:
        import gdown
        print("[DeepfakeModel] Downloading weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, TORCH_WEIGHTS, quiet=False)
        print(f"[DeepfakeModel] Weights downloaded to {TORCH_WEIGHTS}")
    except ImportError:
        print("[DeepfakeModel] gdown not installed. To auto-download, run: pip install gdown")
    except Exception as e:
        print("[DeepfakeModel] ERROR downloading weights:", e)

# ----------------------------
# Optional ML runtime
# ----------------------------
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ----------------------------
# Torch Xception deepfake wrapper
# ----------------------------
if TORCH_AVAILABLE:
    import torch.nn as nn
    import torchvision.transforms as T

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
        """
        A compact Xception-like architecture. Note: if your checkpoint uses a differently
        named / structured Xception, loading may still produce missing/unexpected keys.
        We will attempt to load with strict=False and print mismatches.
        """
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

    class TorchDeepfakeWrapper:
        def __init__(self, weights_path=TORCH_WEIGHTS, input_size=(160,160), device=None):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(self.device)
            self.input_size = input_size
            self.model = XceptionDeepfake().to(self.device)

            if os.path.exists(weights_path):
                try:
                    # Try normal load first (modern PyTorch might enforce weights_only)
                    try:
                        ckpt = torch.load(weights_path, map_location=self.device)
                    except TypeError:
                        # Older/newer PyTorch may accept weights_only flag - try alternate signature
                        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)

                    # ckpt might be:
                    # - a state_dict (OrderedDict)
                    # - a dict containing 'state_dict' or 'model'
                    state_dict = None
                    if isinstance(ckpt, dict):
                        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                            state_dict = ckpt["state_dict"]
                            print("[DeepfakeModel] checkpoint contains 'state_dict' key")
                        elif "model" in ckpt and isinstance(ckpt["model"], dict):
                            state_dict = ckpt["model"]
                            print("[DeepfakeModel] checkpoint contains 'model' key")
                        else:
                            # maybe it's directly the state dict
                            # check if keys look like state dict keys
                            sample_keys = list(ckpt.keys())[:5]
                            if any(isinstance(k, str) for k in sample_keys):
                                state_dict = ckpt
                            else:
                                state_dict = ckpt  # fallback
                    else:
                        # ckpt is not a dict (unexpected)
                        state_dict = None

                    if state_dict is None:
                        print("[DeepfakeModel] Could not interpret checkpoint format; skipping weight load.")
                    else:
                        # If keys are prefixed with 'model.' remove it to match our layer names
                        first_key = next(iter(state_dict.keys()))
                        if first_key.startswith("model."):
                            new_sd = {}
                            for k,v in state_dict.items():
                                new_sd[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
                            state_dict = new_sd
                            print("[DeepfakeModel] Stripped leading 'model.' prefix from state_dict keys.")

                        # Attempt load with strict=False to avoid crashes; show missing/unexpected keys
                        missing, unexpected = self._load_state_dict_loose(state_dict)
                        if missing or unexpected:
                            print("[DeepfakeModel] load_state_dict results:")
                            if missing:
                                print("  Missing keys:", missing[:10], "..." if len(missing)>10 else "")
                            if unexpected:
                                print("  Unexpected keys:", unexpected[:10], "..." if len(unexpected)>10 else "")
                        else:
                            print("[DeepfakeModel] Weights loaded successfully (no missing/unexpected keys).")

                except Exception as e:
                    print("[DeepfakeModel] Failed to load weights:", e)
            else:
                print("[DeepfakeModel] WARNING: weights file not found at", weights_path)

            self.model.eval()
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

        def _load_state_dict_loose(self, sd: dict):
            """
            Attempts to load sd into self.model with strict=False and returns (missing_keys, unexpected_keys).
            """
            try:
                load_result = self.model.load_state_dict(sd, strict=False)
                # load_result is a NamedTuple (missing_keys, unexpected_keys) in newer PyTorch,
                # older versions may return None. Normalize:
                if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
                    missing = list(load_result.missing_keys)
                    unexpected = list(load_result.unexpected_keys)
                elif isinstance(load_result, dict):
                    missing = load_result.get("missing_keys", []) or []
                    unexpected = load_result.get("unexpected_keys", []) or []
                else:
                    # unknown return, assume success
                    missing, unexpected = [], []
                return missing, unexpected
            except Exception as e:
                # final fallback: try to load keys one-by-one where possible
                print("[DeepfakeModel] load_state_dict raised:", e)
                return ["error_loading"], ["error_loading"]

        def infer(self, frames):
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

else:
    TorchDeepfakeWrapper = None

# ---------------------------
# Audio Analyzer
# ---------------------------
class AudioAnalyzer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.sample_rate = 16000
        self.frames = int(self.sample_rate * 0.5)
        self.score_history = deque(maxlen=6)
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while self.running:
                audio_chunk, overflowed = stream.read(self.frames)
                block = audio_chunk.flatten()
                block /= (np.max(np.abs(block)) + 1e-9)

                flat = self.spectral_flatness(block)
                peaks = self.spectral_peaks(block)
                energy = np.std(block)

                flat_score = np.clip((flat - 0.25)/0.75, 0, 1)
                peak_score = np.clip(1 - peaks/45, 0, 1)
                stable_score = np.clip(1 - (energy*25), 0, 1)

                audio_score = 0.45*flat_score + 0.35*peak_score + 0.2*stable_score

                with self.lock:
                    self.score_history.append(audio_score)

    def spectral_flatness(self, x):
        S = np.abs(np.fft.rfft(x)) + 1e-12
        return np.exp(np.mean(np.log(S))) / np.mean(S)

    def spectral_peaks(self, x):
        S = 20*np.log10(np.abs(np.fft.rfft(x)) + 1e-12)
        peaks,_ = find_peaks(S, height=np.max(S)*0.1)
        return len(peaks)

    def get_score(self):
        with self.lock:
            if not self.score_history:
                return 0
            return float(np.mean(self.score_history))

    def stop(self):
        self.running = False

# ---------------------------
# Video Analyzer
# ---------------------------
class VideoAnalyzer(threading.Thread):
    def __init__(self, model_wrapper=None):
        super().__init__(daemon=True)
        self.running = False
        self.frame_history = deque(maxlen=20)
        self.score_history = deque(maxlen=6)
        self.lock = threading.Lock()
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_buffer = []
        self.model_wrapper = model_wrapper
        self.last_model_infer = 0
        self.model_score = 0

    def run(self):
        self.running = True
        with mss.mss() as sct:  # MSS inside thread
            while self.running:
                frame = np.asarray(sct.grab(sct.monitors[1]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    time.sleep(0.1)
                    continue
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                face_color = frame[y:y+h, x:x+w]
                face_gray = cv2.resize(gray[y:y+h, x:x+w], (160,160))
                self.frame_history.append(face_gray)

                # heuristic
                repeat_score, motion_score, moire_score = 0,0,0
                if len(self.frame_history) > 3:
                    arr = list(self.frame_history)
                    diffs = [np.mean(np.abs(arr[i]-arr[i-1])) for i in range(1,len(arr))]
                    small_diffs = sum(1 for d in diffs if d < 4.0)
                    repeat_score = np.clip(small_diffs/6.0, 0, 1)
                    try:
                        flow = cv2.calcOpticalFlowFarneback(arr[-2], arr[-1], None,0.5,3,15,3,5,1.2,0)
                        mag,_ = cv2.cartToPolar(flow[...,0],flow[...,1])
                        motion_score = np.clip((30-np.var(mag))/30,0,1)
                    except:
                        pass
                    lap = cv2.Laplacian(arr[-1],cv2.CV_64F).var()
                    moire_score = np.clip((lap-150)/150,0,1)
                video_score = 0.45*repeat_score + 0.35*motion_score + 0.2*moire_score
                with self.lock:
                    self.score_history.append(video_score)

                # model buffer
                if self.model_wrapper:
                    face_model = cv2.resize(face_color, (160,160))
                    self.face_buffer.append(face_model)
                    if len(self.face_buffer) > 32:
                        self.face_buffer = self.face_buffer[-32:]
                    if time.time()-self.last_model_infer > 1.0 and len(self.face_buffer) > 3:
                        sample = [self.face_buffer[i] for i in np.linspace(0,len(self.face_buffer)-1, min(len(self.face_buffer),16)).astype(int)]
                        try:
                            probs = self.model_wrapper.infer(sample)
                            self.model_score = np.mean(probs)
                        except:
                            self.model_score = 0
                        self.last_model_infer = time.time()

                time.sleep(0.1)

    def get_score(self):
        with self.lock:
            if not self.score_history:
                return 0
            return float(np.mean(self.score_history))

    def get_model_score(self):
        return getattr(self, 'model_score', 0)

    def stop(self):
        self.running = False

# ---------------------------
# Controller
# ---------------------------
class FakeVCDetector:
    def __init__(self):
        model_wrapper = TorchDeepfakeWrapper() if TORCH_AVAILABLE else None
        self.audio = AudioAnalyzer()
        self.video = VideoAnalyzer(model_wrapper=model_wrapper)
        self.running = False

    def start(self):
        self.running = True
        self.audio.start()
        self.video.start()

    def stop(self):
        self.running = False
        self.audio.stop()
        self.video.stop()

    def get_scores(self):
        a = self.audio.get_score()
        v = self.video.get_score()
        m = self.video.get_model_score()
        fused = (0.3*a + 0.3*v + 0.4*m) if TORCH_AVAILABLE else (0.5*a + 0.5*v)
        return {"audio":a, "video":v, "model":m, "fused":fused}

    def get_label(self):
        fused = self.get_scores()["fused"]
        if fused > 0.65:
            return "LIKELY FAKE (HIGH CONFIDENCE)"
        elif fused > 0.45:
            return "POSSIBLE FAKE"
        else:
            return "LIKELY REAL"
