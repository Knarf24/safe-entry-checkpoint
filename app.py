"""Safe Entry Checkpoint: a door camera that checks masks, counts people and speaks the result."""

from __future__ import annotations

from datetime import datetime, timezone

import gradio as gr
import numpy as np

from checkpoint import MaskChecker, decide, draw, speak

checker = MaskChecker()

INTRO = """
# 🚪 Safe Entry Checkpoint
A door camera for clinics, shops and offices. It **finds every face**, **checks each one for a mask**,
**keeps a count of how many people are inside**, and **says the result out loud**, so staff at the door
don't have to do it all by hand.

Take a photo with your webcam (or upload one), then press **Check entry**.
"""

ABOUT = """
### How it works
1. **Face detection and counting:** OpenCV's YuNet face detector finds every face in the photo.
2. **Mask check:** each face goes through a MobileNetV2 network, and a small classifier trained on
   about 4,000 photos of people with and without masks decides *mask* or *no mask*
   (about 99% accurate in testing). A mask pulled down under the chin counts as no mask.
3. **Decision:** everyone must wear a mask (if required) and there must be room left inside.
   People who pass are added to the count. Press **Someone left** when a person leaves.
4. **Speech:** the result is spoken with Google Text-to-Speech (gTTS).

Photos are processed in memory and never stored. Built in Python with OpenCV, NumPy, Gradio and gTTS.
"""

LOG_HEADERS = ["Time (UTC)", "People", "Masks", "Result", "Inside after"]


def occupancy_md(inside: int, capacity: int) -> str:
    pct = 0 if capacity <= 0 else min(100, round(100 * inside / capacity))
    color = "#2ea043" if pct < 80 else "#d29922" if pct < 100 else "#da3633"
    return (
        f"<div style='font-size:1.1rem'><b>Inside now: {inside} / {capacity}</b></div>"
        f"<div style='background:#8884;border-radius:6px;height:14px;margin-top:6px'>"
        f"<div style='width:{pct}%;background:{color};height:14px;border-radius:6px'></div></div>"
    )


def banner(allowed: bool | None, headline: str) -> str:
    if allowed is None:
        bg, icon = "#6e7681", "📷"
    elif allowed:
        bg, icon = "#2ea043", "✅"
    else:
        bg, icon = "#da3633", "⛔"
    return (
        f"<div style='background:{bg};color:white;padding:18px;border-radius:10px;"
        f"font-size:1.4rem;font-weight:700;text-align:center'>{icon} {headline}</div>"
    )


def check(image, inside, capacity, mask_required, voice, log):
    capacity = int(capacity)
    if image is None:
        d_headline, speech = "Take or upload a photo first", "Please step in front of the camera."
        return None, banner(None, d_headline), occupancy_md(inside, capacity), inside, None, speech, False, log, log
    rgb = np.asarray(image)[:, :, :3]
    faces = checker.analyse(rgb)
    d = decide(faces, inside, capacity, mask_required)
    if d.allowed:
        inside += len(faces)
    masked = sum(f.masked for f in faces)
    row = [
        datetime.now(timezone.utc).strftime("%H:%M:%S"),
        len(faces),
        f"{masked}/{len(faces)}",
        "Allowed" if d.allowed else "Stopped",
        inside,
    ]
    log = [row] + list(log or [])
    audio = speak(d.speech) if voice else None
    return (
        draw(rgb, faces, mask_required),
        banner(d.allowed, d.headline),
        occupancy_md(inside, capacity),
        inside,
        audio,
        d.speech,
        bool(audio) or not voice,  # True means "no browser fallback needed"
        log,
        log,
    )


def someone_left(inside, capacity):
    inside = max(0, inside - 1)
    return inside, occupancy_md(inside, int(capacity))


def reset(capacity):
    return 0, occupancy_md(0, int(capacity)), [], [], banner(None, "Ready")


# If gTTS can't reach Google, fall back to the browser's own voice.
BROWSER_SPEAK = """
(text, done) => {
  if (!done && text && window.speechSynthesis) {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
}
"""

with gr.Blocks(title="Safe Entry Checkpoint") as demo:
    inside_state = gr.State(0)
    speech_state = gr.State("")
    spoken_state = gr.State(True)
    log_state = gr.State([])

    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            cam = gr.Image(
                label="Door camera",
                sources=["webcam", "upload"],
                type="numpy",
                height=380,
            )
            check_btn = gr.Button("Check entry", variant="primary", size="lg")
            with gr.Accordion("Settings", open=False):
                capacity = gr.Slider(1, 200, value=20, step=1, label="Maximum people inside")
                mask_required = gr.Checkbox(True, label="Masks required")
                voice = gr.Checkbox(True, label="Speak the result")
        with gr.Column(scale=1):
            result_banner = gr.HTML(banner(None, "Ready"))
            annotated = gr.Image(label="What the camera saw", interactive=False, height=380)
            audio = gr.Audio(label="Spoken result", autoplay=True, interactive=False)
            occ = gr.HTML(occupancy_md(0, 20))
            with gr.Row():
                left_btn = gr.Button("Someone left (−1)")
                reset_btn = gr.Button("Reset count")
    log = gr.Dataframe(headers=LOG_HEADERS, label="Entry log", interactive=False, value=[])
    gr.Markdown(ABOUT)

    check_btn.click(
        check,
        [cam, inside_state, capacity, mask_required, voice, log_state],
        [annotated, result_banner, occ, inside_state, audio, speech_state, spoken_state, log_state, log],
    ).then(None, [speech_state, spoken_state], None, js=BROWSER_SPEAK)
    left_btn.click(someone_left, [inside_state, capacity], [inside_state, occ])
    reset_btn.click(reset, [capacity], [inside_state, occ, log_state, log, result_banner])
    capacity.change(lambda i, c: occupancy_md(i, int(c)), [inside_state, capacity], occ)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
