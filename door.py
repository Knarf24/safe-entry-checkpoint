"""The door logic the web page calls. Runs in the browser through Pyodide.

The page sends each photo here; Python finds the faces, checks the masks,
keeps the count of people inside, keeps the entry log and speaks the result.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from checkpoint import MaskChecker, decide, draw, speak

checker = MaskChecker()
inside = 0
log: list[dict] = []


def check(rgba, width: int, height: int, capacity: int, mask_required: bool, voice: bool):
    """rgba: the photo's pixels from a canvas. Returns (annotated RGBA bytes, result JSON)."""
    global inside
    pixels = np.frombuffer(rgba.to_py() if hasattr(rgba, "to_py") else rgba, np.uint8)
    rgb = pixels.reshape(height, width, 4)[:, :, :3].copy()

    faces = checker.analyse(rgb)
    d = decide(faces, inside, int(capacity), bool(mask_required))
    if d.allowed:
        inside += len(faces)
    masked = sum(f.masked for f in faces)
    log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "people": len(faces),
            "masks": f"{masked}/{len(faces)}",
            "result": "Allowed" if d.allowed else "Stopped",
            "inside": inside,
        },
    )
    if voice:
        speak(d.speech)

    annotated = draw(rgb, faces, bool(mask_required))
    out = np.dstack([annotated, np.full(annotated.shape[:2], 255, np.uint8)])
    result = {
        "allowed": d.allowed,
        "headline": d.headline,
        "speech": d.speech,
        "people": len(faces),
        "masked": masked,
        "inside": inside,
        "log": log[:50],
        "faces": [
            {"box": list(f.box), "mask": round(f.mask_prob, 3)} for f in faces
        ],
    }
    return out.tobytes(), json.dumps(result)


def someone_left() -> int:
    global inside
    inside = max(0, inside - 1)
    return inside


def reset() -> int:
    global inside
    inside = 0
    log.clear()
    return inside
