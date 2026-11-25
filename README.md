---
title: Safe Entry Checkpoint
emoji: 🚪
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Door camera that checks masks, counts people and speaks
---

# 🚪 Safe Entry Checkpoint

**The problem:** staff at the door of a clinic, shop or office can't check masks and keep an
occupancy count at the same time.

**The app:** a door camera that finds every face, checks each one for a mask, keeps count of how
many people are inside, and says the result out loud.

Built in **Python** by combining four GeeksforGeeks projects:

| GeeksforGeeks project | What it does here |
|---|---|
| [Face Mask Detection](https://www.geeksforgeeks.org/python/face-mask-detection-and-thermal-scanner-for-covid-care-python-project/) | Decides *mask* or *no mask* for each face (a mask under the chin counts as no mask) |
| [Face Detection with Webcam](https://www.geeksforgeeks.org/python/face-detection-using-python-and-opencv-with-webcam/) | Finds faces in the webcam photo with OpenCV |
| [Count Number of Faces](https://www.geeksforgeeks.org/python/count-number-of-faces-using-python-opencv/) | Counts the people at the door and keeps the occupancy count |
| [Convert Text to Speech](https://www.geeksforgeeks.org/python/convert-text-speech-python/) | Speaks the result with gTTS |

## How it works

1. **Face detection:** OpenCV's YuNet detector (`cv2.FaceDetectorYN`) finds every face.
2. **Mask check:** each face crop goes through MobileNetV2 (OpenCV DNN). A small logistic-regression
   head trained on about 4,000 photos decides mask or no mask (99% in 5-fold cross-validation).
3. **Decision:** everyone must wear a mask (can be switched off) and there must be room inside.
   People who pass are added to the count; **Someone left** takes one off.
4. **Speech:** the result is spoken with gTTS, with the browser's voice as a fallback.

Photos are processed in memory and never stored.

## Run it yourself

```bash
pip install gradio -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860.

## Files

- `app.py`: the Gradio web page
- `checkpoint.py`: face detection, mask check, decision and speech
- `train_mask_head.py`: re-trains the mask classifier
- `models/`: YuNet face detector, MobileNetV2 and the trained mask head
- `.github/workflows/sync-to-hf.yml`: every push to `main` deploys to Hugging Face Spaces

## Credits

- Face detector: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) from OpenCV Zoo (MIT)
- Backbone: [MobileNetV2](https://github.com/opencv/opencv_zoo/tree/main/models/image_classification_mobilenet) from OpenCV Zoo (Apache 2.0)
- Training photos: [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) (MIT)
