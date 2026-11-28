# 🚪 Safe Entry Checkpoint

**Live demo:** https://knarf24.github.io/safe-entry-checkpoint/

**The problem:** staff at the door of a clinic, shop or office can't check masks and keep an
occupancy count at the same time.

**The app:** a door camera that finds every face, checks each one for a mask, keeps count of how
many people are inside, and says the result out loud.

It is written in **Python** (OpenCV + NumPy) and runs **inside your browser** with
[Pyodide](https://pyodide.org), so it is hosted free on GitHub Pages and photos never leave your device.

Built by combining four GeeksforGeeks Python projects:

| GeeksforGeeks project | What it does here |
|---|---|
| [Face Mask Detection](https://www.geeksforgeeks.org/python/face-mask-detection-and-thermal-scanner-for-covid-care-python-project/) | Decides *mask* or *no mask* for each face (a mask under the chin counts as no mask) |
| [Face Detection with Webcam](https://www.geeksforgeeks.org/python/face-detection-using-python-and-opencv-with-webcam/) | Finds faces in the webcam photo with OpenCV |
| [Count Number of Faces](https://www.geeksforgeeks.org/python/count-number-of-faces-using-python-opencv/) | Counts the people at the door and keeps the occupancy count |
| [Convert Text to Speech](https://www.geeksforgeeks.org/python/convert-text-speech-python/) | Speaks the result (browser voice in the web app, gTTS on a normal Python install) |

## How it works

1. **Face detection:** OpenCV's YuNet detector (`cv2.FaceDetectorYN`) finds every face.
2. **Mask check:** each face crop goes through MobileNetV2 (OpenCV DNN). A small logistic-regression
   head trained on about 4,000 photos decides mask or no mask (99% in 5-fold cross-validation).
3. **Decision:** everyone must wear a mask (can be switched off) and there must be room inside.
   People who pass are added to the count; **Someone left** takes one off.
4. **Speech:** Python calls the browser's built-in voice to say the result.

## Files

- `checkpoint.py`: face detection, mask check, decision and speech (plain Python, also runs on a normal install)
- `door.py`: the checkpoint state (count and log) that the web page calls
- `web/`: the page (`index.html`, `main.js` for the camera, `style.css`) and the Pyodide runtime with OpenCV
- `models/`: YuNet face detector, MobileNetV2 and the trained mask head
- `train_mask_head.py`: re-trains the mask classifier
- `build_site.sh` and `.github/workflows/pages.yml`: every push to `main` publishes to GitHub Pages

## Run it on your computer

```bash
sh build_site.sh
cd _site && python -m http.server 8000
```

Then open http://localhost:8000.

## Fast loading

The page shows straight away and Python starts in the background, so nothing waits on a
"Starting Python" screen. Anything that needs Python simply waits for it if you are quicker.

- **Saved on the device** (`web/sw.js`): after the first visit, Python (Pyodide) and its packages are kept in
  the browser, so a refresh or a later visit doesn't download them again. Every app on the same site shares
  that one saved copy. The app's own files are still fetched fresh, so updates show up right away.
- **Quick restart** (`web/pyfast.js`): the first visit also saves a snapshot of a freshly started Python.
  Later visits restore it instead of starting Python from scratch (about 1.2 s down to 0.3 s on a laptop).
  If the snapshot can't be used (for example after a Pyodide upgrade), Python starts normally and a new one
  is saved.
- `build_site.sh` writes the Pyodide version into `sw.js`, so upgrading `web/pyodide/` needs no other change.


## Credits

- [Pyodide](https://github.com/pyodide/pyodide) 0.29.5 with its OpenCV and NumPy builds (MPL 2.0 and the packages' own licenses)
- Face detector: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) from OpenCV Zoo (MIT)
- Backbone: [MobileNetV2](https://github.com/opencv/opencv_zoo/tree/main/models/image_classification_mobilenet) from OpenCV Zoo (Apache 2.0)
- Training photos: [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) (MIT)
