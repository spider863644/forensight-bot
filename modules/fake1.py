# fake_vc_detector.py
"""
Cross-Platform Real-Time Fake Video Call Detector (Audio + Video + Deepfake Model)
Works on: Windows / macOS / Linux

Dependencies (suggested):
  pip install numpy opencv-python mss soundcard scipy torch torchvision onnxruntime

Behavior:
  - Captures system (speaker) audio via soundcard (loopback)
  - Captures full screen via mss, detects the largest face, crops it
  - Runs lightweight video heuristics (repeated frames, low micro-motion, moire)
  - Buffers face crops and runs a deepfake classifier (if available)
  - Fuses audio_score + heuristic_video_score + model_score into a final label printed live

How to use:
  python fake_vc_detector.py
"""

import time
import threading
from collections import deque
import numpy as np
import cv2
import mss
import soundcard as sc
from scipy.signal import find_peaks
from scipy import fftpack
import os
import sys
import math
import traceback

# Optional ML runtimes (import if available)
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# ----------------------------
# CONFIGURATION
# ----------------------------
AUDIO_SAMPLE_RATE = 16000
AUDIO_BLOCK_SECONDS = 0.6
AUDIO_FRAMES = int(AUDIO_SAMPLE_RATE * AUDIO_BLOCK_SECONDS)
AUDIO_HISTORY = 10

VIDEO_CAPTURE_FPS = 8
FACE_BUFFER_MAX = 32            # how many face crops to buffer before model inference
MODEL_INFER_INTERVAL = 1.0     # seconds between model inference rounds
SCORE_WINDOW = 6

# Heuristic thresholds (tune for your environment)
SCORE_THRESH_HIGH = 0.65
SCORE_THRESH_MED = 0.45

# Deepfake model config
MODEL_CFG = {
    "enabled": True,
    # prefer PyTorch if available; set PATH to your checkpoint if you have one
    "torch_weights_path": "models/xception_ffpp.pth",
    "onnx_path": "models/deepfake_detector.onnx",
    "input_size": (160, 160),  # face crop resize for model
    "device": "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
}

# Weights for fusion (heuristics + model)
FUSION_WEIGHTS = {
    "audio": 0.3,
    "video_heuristic": 0.3,
    "model": 0.4
}

# ----------------------------
# Utility functions
# ----------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x.astype(np.float64))))

def spectral_flatness(sig):
    eps = 1e-12
    S = np.abs(np.fft.rfft(sig)) + eps
    geo = np.exp(np.mean(np.log(S)))
    arith = np.mean(S)
    return float(geo / (arith + 1e-12))

def spectral_peaks_count(sig):
    S = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-12)
    peaks, _ = find_peaks(S, height=np.max(S)*0.12 if S.size>0 else 0.0)
    return len(peaks)

def frame_diff_score(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def laplacian_var(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

# ----------------------------
# Optional lightweight PyTorch model wrapper (Xception-like placeholder)
# You should replace this with your production Xception model implementation and weights.
# ----------------------------
class TorchDeepfakeWrapper:
    def __init__(self, weights_path, input_size=(160,160), device="cpu"):
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # minimal model definition if weights not available; replace in prod with Xception
        class MiniNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
                    nn.Conv2d(32,64,3,padding=1,stride=2), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64,1)
                )
            def forward(self, x):
                return self.net(x)

        try:
            self.model = MiniNet().to(self.device)
            if os.path.exists(weights_path):
                ckpt = torch.load(weights_path, map_location=self.device)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                elif isinstance(ckpt, dict):
                    try:
                        self.model.load_state_dict(ckpt)
                    except Exception:
                        # maybe full model saved
                        self.model = ckpt.to(self.device)
                else:
                    self.model = ckpt.to(self.device)
                print("[DeepfakeModel] Loaded PyTorch weights from:", weights_path)
            else:
                print("[DeepfakeModel] PyTorch weights not found, using random-init MiniNet (fallback).")
            self.model.eval()
        except Exception as e:
            print("[DeepfakeModel] Torch model init error:", e)
            self.model = None

    def infer(self, frames):
        """frames: list of BGR numpy arrays. Returns list of prob(fake) floats."""
        if self.model is None:
            # fallback: return neutral low scores
            return [0.05 for _ in frames]
        with torch.no_grad():
            tensors = []
            for f in frames:
                try:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    t = self.transform(rgb)
                    tensors.append(t)
                except Exception:
                    # create zero tensor to keep shapes consistent
                    tensors.append(torch.zeros(3, self.input_size[0], self.input_size[1]))
            x = torch.stack(tensors).to(self.device)
            logits = self.model(x)
            # handle shapes
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().astype(float).tolist()
            return probs

# ----------------------------
# Optional ONNX wrapper
# ----------------------------
class ONNXDeepfakeWrapper:
    def __init__(self, onnx_path, input_size=(160,160), providers=None):
        self.input_size = input_size
        self.session = None
        try:
            if providers is None:
                providers = ["CPUExecutionProvider"]
                # if GPU present, let onnxruntime choose sensible provider
                # user can configure env to use CUDAExecutionProvider where available
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            # attempt to find input name
            self.input_name = self.session.get_inputs()[0].name
            print("[DeepfakeModel] ONNX session loaded:", onnx_path)
        except Exception as e:
            print("[DeepfakeModel] ONNX init error:", e)
            self.session = None

    def preprocess(self, frame):
        # expects BGR -> convert to RGB and normalize to [0,1], channel-first
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.input_size[1], self.input_size[0]))
        arr = img.astype(np.float32) / 255.0
        # normalize (same mean/std as Torch wrapper)
        mean = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(1,1,3)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(1,1,3)
        arr = (arr - mean) / std
        # channel first
        arr = np.transpose(arr, (2,0,1)).astype(np.float32)
        return arr

    def infer(self, frames):
        if self.session is None:
            return [0.05 for _ in frames]
        # prepare batch
        batch = np.stack([self.preprocess(f) for f in frames]).astype(np.float32)
        try:
            outputs = self.session.run(None, {self.input_name: batch})
            # assume outputs[0] contains logits for fake class, shape [B,1] or [B]
            logits = outputs[0]
            # apply sigmoid
            probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
            return probs.tolist()
        except Exception as e:
            print("[DeepfakeModel] ONNX inference error:", e)
            return [0.05 for _ in frames]

# ----------------------------
# Audio Analyzer (thread)
# ----------------------------
class AudioAnalyzer(threading.Thread):
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE, frames=AUDIO_FRAMES):
        super().__init__(daemon=True)
        self.running = False
        self.sample_rate = sample_rate
        self.frames = frames
        self.history = deque(maxlen=AUDIO_HISTORY)
        self.score_history = deque(maxlen=SCORE_WINDOW)
        self.lock = threading.Lock()
        try:
            self.speaker = sc.default_speaker()
        except Exception as e:
            print("[AudioAnalyzer] soundcard default_speaker error:", e)
            self.speaker = None

    def run(self):
        if self.speaker is None:
            print("[AudioAnalyzer] No speaker available; audio disabled.")
            return
        self.running = True
        try:
            with self.speaker.recorder(samplerate=self.sample_rate, channels=1) as rec:
                while self.running:
                    block = rec.record(numframes=self.frames).flatten().astype(np.float32)
                    if block.size == 0:
                        continue
                    # normalize robustly
                    denom = (np.max(np.abs(block)) + 1e-9)
                    block = block / denom

                    flat = spectral_flatness(block)
                    peaks = spectral_peaks_count(block)
                    energy = np.std(block)

                    flat_score = np.clip((flat - 0.25) / 0.75, 0.0, 1.0)
                    peak_score = np.clip(1.0 - peaks/45.0, 0.0, 1.0)
                    stable_score = np.clip(1.0 - (energy*25.0), 0.0, 1.0)

                    audio_score = float(0.45*flat_score + 0.35*peak_score + 0.2*stable_score)

                    with self.lock:
                        self.score_history.append(audio_score)
        except Exception as e:
            print("[AudioAnalyzer] runtime error:", e)
            traceback.print_exc()
            self.running = False

    def get_score(self):
        with self.lock:
            if not self.score_history:
                return 0.0
            return float(np.mean(self.score_history))

    def stop(self):
        self.running = False

# ----------------------------
# Video Analyzer (cross-platform) + model integration
# ----------------------------
class VideoAnalyzer(threading.Thread):
    def __init__(self, model_wrapper=None, input_size=(160,160)):
        super().__init__(daemon=True)
        self.running = False
        self.sct = mss.mss()
        self.frame_history = deque(maxlen=20)
        self.score_history = deque(maxlen=SCORE_WINDOW)
        self.lock = threading.Lock()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_buffer = []  # color BGR frames of face crops for model
        self.model_wrapper = model_wrapper
        self.input_size = input_size
        self.last_model_infer = 0.0
        self.model_score_history = deque(maxlen=SCORE_WINDOW)

    def run(self):
        self.running = True
        prev_face = None
        prev_gray = None
        while self.running:
            start = time.time()
            try:
                # take full-screen capture (primary monitor)
                monitor = self.sct.monitors[1]
                s = self.sct.grab(monitor)
                frame = np.asarray(s)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                # capture errors may happen on some OS/permission combos
                print("[VideoAnalyzer] screen capture error:", e)
                time.sleep(0.5)
                continue

            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                # no face detected; reduce activity
                time.sleep(max(0.01, (1.0 / VIDEO_CAPTURE_FPS)))
                continue

            # choose largest face (assume remote participant displayed at reasonable size)
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            # add margin
            pad = int(min(w, h) * 0.15)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)

            face_color = frame[y0:y1, x0:x1]
            face_gray = gray[y0:y1, x0:x1]
            # normalize face size for heuristics
            face_small = cv2.resize(face_gray, (160, 160))
            self.frame_history.append(face_small)

            # Heuristics:
            repeat_score = 0.0
            motion_score = 0.0
            moire_score = 0.0

            if len(self.frame_history) > 3:
                arr = list(self.frame_history)
                diffs = [frame_diff_score(arr[i], arr[i-1]) for i in range(1, len(arr))]
                small_diffs = sum(1 for d in diffs if d < 4.0)
                repeat_score = np.clip(small_diffs / 6.0, 0.0, 1.0)

                # optical flow micro-movement
                try:
                    prev_f = arr[-2]
                    curr_f = arr[-1]
                    flow = cv2.calcOpticalFlowFarneback(prev_f, curr_f, None,
                                                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                    motion_var = float(np.var(mag))
                    motion_score = np.clip((30.0 - motion_var) / 30.0, 0.0, 1.0)
                except Exception:
                    motion_score = 0.0

                lap = laplacian_var(face_small)
                moire_score = np.clip((lap - 150.0) / 150.0, 0.0, 1.0)

            heuristic_video_score = float(0.45*repeat_score + 0.35*motion_score + 0.2*moire_score)

            # add heuristic to score history
            with self.lock:
                self.score_history.append(heuristic_video_score)

            # Add color face crop to buffer for model inference
            if self.model_wrapper is not None:
                try:
                    # Resize color crop to model input size
                    face_for_model = cv2.resize(face_color, (self.input_size[1], self.input_size[0]))
                    self.face_buffer.append(face_for_model)
                    # cap buffer size
                    if len(self.face_buffer) > FACE_BUFFER_MAX:
                        self.face_buffer = self.face_buffer[-FACE_BUFFER_MAX:]
                except Exception as e:
                    pass

            # Periodically run model inference on buffered frames
            now = time.time()
            if self.model_wrapper is not None and (now - self.last_model_infer) >= MODEL_INFER_INTERVAL and len(self.face_buffer) > 3:
                try:
                    # choose a subset for efficiency: evenly sample up to 16 frames
                    buf = self.face_buffer
                    n = min(len(buf), 16)
                    idxs = np.linspace(0, len(buf)-1, n).astype(int).tolist()
                    sample_frames = [buf[i] for i in idxs]
                    # run model inference
                    probs = self.model_wrapper.infer(sample_frames)
                    # collect mean probability as model_score
                    model_score = float(np.mean(probs))
                    self.model_score_history.append(model_score)
                    self.last_model_infer = now
                except Exception as e:
                    print("[VideoAnalyzer] model inference error:", e)
                    traceback.print_exc()
                    # proceed without failing

            # push loop sleep to control fps
            elapsed = time.time() - start
            target = max(0.01, (1.0 / VIDEO_CAPTURE_FPS) - elapsed)
            time.sleep(target)

    def get_heuristic_score(self):
        with self.lock:
            if not self.score_history:
                return 0.0
            return float(np.mean(self.score_history))

    def get_model_score(self):
        with self.lock:
            if not self.model_score_history:
                return 0.0
            return float(np.mean(self.model_score_history))

    def stop(self):
        self.running = False

# ----------------------------
# Controller: fuse audio + video + model
# ----------------------------
class RealTimeFakeVC:
    def __init__(self, use_model=True):
        self.use_model = use_model and MODEL_CFG.get("enabled", True)
        self.audio = AudioAnalyzer()
        # instantiate model wrapper if possible
        model_wrapper = None
        if self.use_model:
            # prefer PyTorch
            if TORCH_AVAILABLE:
                try:
                    model_wrapper = TorchDeepfakeWrapper(MODEL_CFG["torch_weights_path"],
                                                         input_size=MODEL_CFG["input_size"],
                                                         device=MODEL_CFG["device"])
                    # if torch model created but uninitialized, it's still ok (fallback)
                except Exception as e:
                    print("[RealTimeFakeVC] Torch model init failed:", e)
                    model_wrapper = None
            # else try ONNX runtime
            if model_wrapper is None and ORT_AVAILABLE and os.path.exists(MODEL_CFG["onnx_path"]):
                try:
                    model_wrapper = ONNXDeepfakeWrapper(MODEL_CFG["onnx_path"], input_size=MODEL_CFG["input_size"])
                except Exception as e:
                    print("[RealTimeFakeVC] ONNX model init failed:", e)
                    model_wrapper = None

            if model_wrapper is None:
                print("[RealTimeFakeVC] Model wrappers unavailable; continuing with heuristics only.")
                self.use_model = False

        self.video = VideoAnalyzer(model_wrapper=model_wrapper, input_size=MODEL_CFG["input_size"])
        self.running = False

    def start(self):
        print("[Forensight] Starting cross-platform RealTimeFakeVC detector.")
        print(" - Capturing screen + system audio. Keep the caller video visible (not minimized).")
        if self.use_model:
            print(" - Deepfake model enabled.")
        else:
            print(" - Model disabled; running heuristics-only mode.")

        self.running = True
        self.audio.start()
        self.video.start()

        try:
            while self.running:
                a_score = self.audio.get_score()
                v_heur = self.video.get_heuristic_score()
                v_model = self.video.get_model_score() if self.use_model else 0.0

                # fuse according to weights
                fused = (FUSION_WEIGHTS["audio"] * a_score +
                         FUSION_WEIGHTS["video_heuristic"] * v_heur +
                         (FUSION_WEIGHTS["model"] * v_model if self.use_model else 0.0))

                # produce label
                if fused >= SCORE_THRESH_HIGH:
                    label = "LIKELY FAKE (HIGH CONFIDENCE)"
                elif fused >= SCORE_THRESH_MED:
                    label = "POSSIBLE FAKE (LOW CONFIDENCE)"
                else:
                    label = "LIKELY REAL"

                # print status (compact)
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] Audio:{a_score:.2f} VideoHeur:{v_heur:.2f} Model:{v_model:.2f} -> {label} (score={fused:.2f})")

                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping detector...")
            self.stop()
        except Exception as e:
            print("Runtime error in controller:", e)
            traceback.print_exc()
            self.stop()

    def stop(self):
        self.running = False
        self.audio.stop()
        self.video.stop()

# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    # Provide a safe, non-privileged run by default
    use_model = MODEL_CFG.get("enabled", True)
    detector = RealTimeFakeVC(use_model=use_model)
    detector.start()

if __name__ == "__main__":
    main()
