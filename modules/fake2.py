"""
Cross-Platform Real-Time Fake Video Call Detector
Windows / macOS / Linux

Audio: soundcard loopback (cross-platform)
Video: screen capture via mss + face detection
"""

import time
import threading
import numpy as np
import cv2
import mss
import soundcard as sc
from scipy.signal import find_peaks
from collections import deque

# ---------------------------
# Audio Analyzer
# ---------------------------
class AudioAnalyzer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.sample_rate = 16000
        self.frames = int(self.sample_rate * 0.5)
        self.history = deque(maxlen=10)
        self.score_history = deque(maxlen=6)
        self.lock = threading.Lock()
        self.speaker = sc.default_speaker()

    def run(self):
        self.running = True
        with self.speaker.recorder(samplerate=self.sample_rate, channels=1) as rec:
            while self.running:
                block = rec.record(numframes=self.frames).flatten().astype(np.float32)
                block /= (np.max(np.abs(block)) + 1e-9)

                flat = self.spectral_flatness(block)
                peaks = self.spectral_peaks(block)
                energy = np.std(block)

                flat_score = np.clip((flat - 0.25) / 0.75, 0, 1)
                peak_score = np.clip(1 - peaks/45, 0, 1)
                stable_score = np.clip(1 - (energy*25), 0, 1)

                audio_score = float(0.45*flat_score + 0.35*peak_score + 0.2*stable_score)

                with self.lock:
                    self.score_history.append(audio_score)

    def spectral_flatness(self, x):
        S = np.abs(np.fft.rfft(x)) + 1e-12
        return np.exp(np.mean(np.log(S))) / np.mean(S)

    def spectral_peaks(self, x):
        S = 20*np.log10(np.abs(np.fft.rfft(x)) + 1e-12)
        peaks, _ = find_peaks(S, height=np.max(S)*0.1)
        return len(peaks)

    def get_score(self):
        if not self.score_history:
            return 0
        return float(np.mean(self.score_history))

    def stop(self):
        self.running = False


# ---------------------------
# Video Analyzer (Cross-Platform)
# ---------------------------
class VideoAnalyzer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.sct = mss.mss()
        self.frame_history = deque(maxlen=20)
        self.score_history = deque(maxlen=6)
        self.lock = threading.Lock()

        # Use lightweight Haar cascade (no heavy GPU)
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def run(self):
        self.running = True
        prev_face = None

        while self.running:
            frame = np.asarray(self.sct.grab(self.sct.monitors[1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                continue

            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            face_region = gray[y:y+h, x:x+w]
            face_small = cv2.resize(face_region, (160, 160))

            self.frame_history.append(face_small)

            # Compute motion / repeated-frame score
            repeat_score = 0
            motion_score = 0
            moire_score = 0

            if len(self.frame_history) > 3:
                diffs = []
                arr = list(self.frame_history)

                for i in range(1, len(arr)):
                    diffs.append(np.mean(np.abs(arr[i] - arr[i-1])))

                small_diffs = sum(1 for d in diffs if d < 4.0)
                repeat_score = np.clip(small_diffs / 6.0, 0, 1)

                # Optical flow micro-movement
                try:
                    prev_f = arr[-2]
                    curr_f = arr[-1]
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_f, curr_f, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                    motion_var = np.var(mag)
                    motion_score = np.clip((30 - motion_var) / 30, 0, 1)
                except:
                    pass

                # Laplacian (moire / screen recording)
                lap = cv2.Laplacian(face_small, cv2.CV_64F).var()
                moire_score = np.clip((lap - 150) / 150, 0, 1)

            video_score = float(0.45*repeat_score + 0.35*motion_score + 0.2*moire_score)

            with self.lock:
                self.score_history.append(video_score)

            time.sleep(0.10)

    def get_score(self):
        if not self.score_history:
            return 0
        return float(np.mean(self.score_history))

    def stop(self):
        self.running = False


# ---------------------------
# Main
# ---------------------------
class FakeVCDetector:
    def __init__(self):
        self.audio = AudioAnalyzer()
        self.video = VideoAnalyzer()
        self.running = False

    def start(self):
        print("[Forensight] Cross-platform real-time fake VC detector started.")
        print("Capturing screen + speaker output...")

        self.running = True
        self.audio.start()
        self.video.start()

        try:
            while self.running:
                a = self.audio.get_score()
                v = self.video.get_score()

                score = (a*0.5) + (v*0.5)

                if score > 0.65:
                    label = "LIKELY FAKE (High Confidence)"
                elif score > 0.45:
                    label = "SUSPECT (Check manually)"
                else:
                    label = "LIKELY REAL"

                print(f"Audio:{a:.2f} Video:{v:.2f} => {label} ({score:.2f})")

                time.sleep(1)

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self.audio.stop()
        self.video.stop()


if __name__ == "__main__":
    FakeVCDetector().start()
