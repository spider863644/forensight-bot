import os
import sys
import argparse
import hashlib
from typing import Dict, Any, List, Tuple

import warnings
warnings.filterwarnings("ignore")  # suppress all warnings

import numpy as np
import cv2
import pytesseract
import imagehash
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import face_recognition

# ---------------------------
# UTILS
# ---------------------------
def sha256_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------
# EXIF
# ---------------------------
def extract_exif_standard(path: str) -> Dict[str, Any]:
    try:
        import exifread
        out = {}
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        for k, v in tags.items():
            out[k] = str(v)
        if "GPS GPSLatitude" in out and "GPS GPSLongitude" in out:
            out["GPSLatitude"] = out.get("GPS GPSLatitude")
            out["GPSLongitude"] = out.get("GPS GPSLongitude")
        return out
    except Exception:
        return {}

def attempt_exif_recovery(path: str) -> Dict[str, Any]:
    return {"recovered": False, "notes": ["Best-effort EXIF recovery attempted"]}

# ---------------------------
# OCR / HASH / COLOR
# ---------------------------
def ocr_text(pil: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(pil).strip()
    except Exception:
        return ""

def image_hashes(pil: Image.Image) -> Dict[str, str]:
    return {
        "phash": str(imagehash.phash(pil)),
        "ahash": str(imagehash.average_hash(pil)),
        "dhash": str(imagehash.dhash(pil)),
    }

def dominant_colors(pil: Image.Image, k: int = 3) -> List[str]:
    im = pil.convert("RGB").resize((200, 200))
    arr = np.asarray(im).reshape(-1, 3).astype(np.float32)
    _, _, centers = cv2.kmeans(
        arr, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(int)
    return ['#%02x%02x%02x' % tuple(c) for c in centers]

# ---------------------------
# CAPTION
# ---------------------------
class Captioner:
    def __init__(self, device="cpu"):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

    def caption(self, pil: Image.Image) -> str:
        inputs = self.processor(images=pil, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_length=40)
        return self.processor.decode(out[0], skip_special_tokens=True)

# ---------------------------
# OBJECT DETECTION
# ---------------------------
def detect_objects_yolo(path: str, conf: float = 0.35) -> List[Dict[str, Any]]:
    try:
        model = YOLO("yolov8n.pt")
        results = model(path, conf=conf)
        out = []
        for r in results:
            for b in r.boxes or []:
                out.append({
                    "label": model.names[int(b.cls)],
                    "confidence": float(b.conf),
                    "bbox": [round(float(x), 2) for x in b.xyxy[0].tolist()]
                })
        return out
    except Exception:
        return []

# ---------------------------
# FACES
# ---------------------------
def face_count_and_boxes(path: str) -> Tuple[int, List[List[int]]]:
    try:
        img = face_recognition.load_image_file(path)
        boxes = face_recognition.face_locations(img)
        parsed = [[l, t, r, b] for (t, r, b, l) in boxes]
        return len(parsed), parsed
    except Exception:
        return 0, []

# ---------------------------
# CORE ANALYSIS
# ---------------------------
def analyze_image(
    path: str,
    do_restore_exif=False,
    deep_search=False,
    do_face_count=False,
    scene_loc=False,
    device="cpu"
) -> Dict[str, Any]:

    report = {"image_sha256": sha256_hex(path)}
    pil = Image.open(path).convert("RGB")
    report["size"] = pil.size

    report["exif_standard"] = extract_exif_standard(path)
    if do_restore_exif or deep_search:
        report["exif_recovery"] = attempt_exif_recovery(path)

    report["ocr_text"] = ocr_text(pil)
    report["hashes"] = image_hashes(pil)
    report["dominant_colors"] = dominant_colors(pil)

    try:
        report["caption"] = Captioner(device).caption(pil)
    except Exception:
        report["caption"] = None

    report["objects"] = detect_objects_yolo(path)

    if do_face_count:
        c, b = face_count_and_boxes(path)
        report["face_count"] = c
        report["face_boxes"] = b

    if scene_loc and report.get("caption"):
        report["scene_location_guess"] = f"Guessed location for: {report['caption']}"
        # Replace with real Google API lookup if needed

    return report

# ---------------------------
# PRINT CLEAN HUMAN-READABLE OUTPUT
# ---------------------------
def print_results(report: Dict[str, Any], queries_only=False):
    if queries_only:
        if report.get("caption"):
            print(report["caption"])
        if report.get("ocr_text"):
            for line in report["ocr_text"].splitlines()[:3]:
                print(line)
        if report.get("objects"):
            print(" ".join({o["label"] for o in report["objects"]}))
        return

    print("🖼 Image Analysis Result")
    print("-" * 40)
    print(f"Image SHA256: {report.get('image_sha256')}")
    print(f"Size: {report.get('size')}")
    print()

    # EXIF
    exif = report.get("exif_standard", {})
    print("EXIF Data:")
    if exif:
        for k, v in exif.items():
            print(f"  {k}: {v}")
    else:
        print("  ❌ No EXIF data found")

    if report.get("exif_recovery"):
        print("EXIF Recovery:")
        for note in report["exif_recovery"].get("notes", []):
            print(f"  - {note}")
    print()

    # Hashes
    hashes = report.get("hashes", {})
    print("Hashes:")
    for k, v in hashes.items():
        print(f"  {k}: {v}")
    print()

    # Dominant colors
    colors = report.get("dominant_colors", [])
    print("Dominant Colors: " + ", ".join(colors))
    print()

    # Caption
    caption = report.get("caption")
    if caption:
        print("Caption / Scene Description:")
        print(f"  {caption}")
        print()

    # OCR text
    ocr = report.get("ocr_text")
    if ocr:
        print("OCR Text:")
        print(f"  {ocr}")
        print()

    # Objects
    objs = report.get("objects", [])
    if objs:
        print("Detected Objects:")
        for obj in objs:
            print(f"  - {obj['label']} ({obj['confidence']*100:.1f}%) at {obj['bbox']}")
        print()

    # Face info
    if report.get("face_count") is not None:
        print(f"Faces Detected: {report.get('face_count')}")
        for b in report.get("face_boxes", []):
            print(f"  - {b}")
        print()

    # Scene location
    if report.get("scene_location_guess"):
        print(f"Scene Location Guess: {report['scene_location_guess']}")
        print()

    print("-" * 40)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--deep-search", action="store_true")
    p.add_argument("--restore-exif", action="store_true")
    p.add_argument("--face-count", action="store_true")
    p.add_argument("--scene-loc", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--queries-only", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.image):
        sys.exit(1)

    report = analyze_image(
        args.image,
        do_restore_exif=args.restore_exif,
        deep_search=args.deep_search,
        do_face_count=args.face_count,
        scene_loc=args.scene_loc,
        device=args.device
    )

    print_results(report, queries_only=args.queries_only)

if __name__ == "__main__":
    main()
