"""Core logic for Safe Entry Checkpoint: find faces, check masks, decide, speak.

Combines four GeeksforGeeks Python projects:
  * Face detection with OpenCV        -> YuNet face detector (cv2.FaceDetectorYN)
  * Count number of faces             -> len(faces)
  * Face mask detection               -> MobileNetV2 features + trained mask head
  * Convert text to speech            -> browser voice (in the web app) or gTTS

Runs in the browser through Pyodide, and also on a normal Python install.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

GREEN = (46, 160, 67)  # RGB, images are RGB throughout
RED = (220, 53, 69)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float  # face detector confidence
    mask_prob: float  # 0..1, chance this face wears a mask

    @property
    def masked(self) -> bool:
        return self.mask_prob >= 0.5


@dataclass
class Decision:
    allowed: bool
    headline: str
    speech: str
    faces: list[Face] = field(default_factory=list)


class MaskChecker:
    def __init__(self, face_threshold: float = 0.7):
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"),
            "",
            (320, 320),
            score_threshold=face_threshold,
            nms_threshold=0.3,
            top_k=50,
        )
        self.backbone = cv2.dnn.readNetFromONNX(
            os.path.join(MODELS, "image_classification_mobilenetv2_2022apr.onnx")
        )
        names = self.backbone.getLayerNames()
        self.feature_layer = next(n for n in names if "GlobalAveragePool" in n)
        head = np.load(os.path.join(MODELS, "mask_head.npz"))
        self.w, self.b = head["w"], float(head["b"])

    # ---- face detection + counting -------------------------------------
    def find_faces(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        scale = min(1.0, 960 / max(h, w))  # keep big photos fast
        small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.detector.detect(small)
        out = []
        for d in dets if dets is not None else []:
            x, y, bw, bh = (d[:4] / scale).astype(int)
            out.append(((int(x), int(y), int(bw), int(bh)), float(d[-1])))
        return out

    # ---- mask classification ---------------------------------------------
    def mask_probability(self, rgb: np.ndarray, box) -> float:
        x, y, w, h = box
        H, W = rgb.shape[:2]
        pad = 0.15  # a little context around the face, like the training crops
        x0, y0 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
        x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad)))
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        img = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        self.backbone.setInput(img.transpose(2, 0, 1)[None])
        feat = self.backbone.forward(self.feature_layer).reshape(-1)
        return float(1 / (1 + np.exp(-(feat @ self.w + self.b))))

    def analyse(self, rgb: np.ndarray) -> list[Face]:
        return [Face(box, s, self.mask_probability(rgb, box)) for box, s in self.find_faces(rgb)]


def decide(faces: list[Face], inside: int, capacity: int, mask_required: bool) -> Decision:
    n = len(faces)
    unmasked = sum(not f.masked for f in faces)
    space = max(0, capacity - inside)
    people = "person" if n == 1 else "people"
    if n == 0:
        return Decision(False, "No one detected", "Please step in front of the camera.", faces)
    if mask_required and unmasked:
        who = "you" if n == 1 else f"{unmasked} of you"
        return Decision(
            False,
            f"Mask needed: {unmasked} of {n} without a mask",
            f"Sorry, {who} must wear a mask before entering.",
            faces,
        )
    if n > space:
        if space == 0:
            msg = "Sorry, we are full right now. Please wait a moment."
        else:
            msg = f"Sorry, only {space} more {'person' if space == 1 else 'people'} can enter right now."
        return Decision(False, f"Over capacity: {n} {people}, {space} space left", msg, faces)
    left = space - n
    thanks = "Thank you for wearing a mask. " if mask_required else ""
    return Decision(
        True,
        f"Welcome: {n} {people} may enter",
        f"{thanks}Welcome in. {left} {'space' if left == 1 else 'spaces'} left.",
        faces,
    )


def draw(rgb: np.ndarray, faces: list[Face], mask_required: bool) -> np.ndarray:
    out = rgb.copy()
    t = max(2, round(max(out.shape[:2]) / 400))
    for f in faces:
        x, y, w, h = f.box
        ok = f.masked or not mask_required
        color = GREEN if ok else RED
        cv2.rectangle(out, (x, y), (x + w, y + h), color, t)
        label = f"{'Mask' if f.masked else 'No mask'} {round(100 * (f.mask_prob if f.masked else 1 - f.mask_prob))}%"
        fs = max(0.5, w / 220)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        ty = max(th + 8, y)
        cv2.rectangle(out, (x, ty - th - 8), (x + tw + 8, ty), color, -1)
        cv2.putText(out, label, (x + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
    return out


def speak(text: str, lang: str = "en") -> str | None:
    """Say the result out loud.

    In the browser (Pyodide) Python calls the browser's built-in voice directly.
    On a normal computer it uses gTTS and returns the path of an mp3 file.
    """
    if sys.platform == "emscripten":
        import js  # Pyodide bridge to the web page

        js.window.speechSynthesis.cancel()
        utterance = js.SpeechSynthesisUtterance.new(text)
        utterance.lang = "en-US" if lang == "en" else lang
        js.window.speechSynthesis.speak(utterance)
        return None
    try:
        from gtts import gTTS

        path = os.path.join(tempfile.gettempdir(), f"sec_{abs(hash(text))}.mp3")
        if not os.path.exists(path):
            gTTS(text=text, lang=lang).save(path)
        return path
    except Exception:
        return None
