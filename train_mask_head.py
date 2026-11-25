"""Re-train the mask classifier (models/mask_head.npz).

Dataset: https://github.com/chandrikadeb7/Face-Mask-Detection (MIT), folder dataset/
with sub-folders with_mask/ and without_mask/ (about 4,000 face photos).

    git clone --depth 1 https://github.com/chandrikadeb7/Face-Mask-Detection fmd
    pip install scikit-learn
    python train_mask_head.py fmd/dataset
"""

import glob
import os
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from checkpoint import MODELS, MaskChecker, _MEAN, _STD


def features(checker: MaskChecker, path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    checker.backbone.setInput(((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None])
    return checker.backbone.forward(checker.feature_layer).reshape(-1)


def main(root: str):
    checker = MaskChecker()
    X, y = [], []
    for label, folder in [(1, "with_mask"), (0, "without_mask")]:
        for f in sorted(glob.glob(os.path.join(root, folder, "*"))):
            v = features(checker, f)
            if v is not None:
                X.append(v)
                y.append(label)
    X, y = np.array(X), np.array(y)
    model = LogisticRegression(C=0.1, max_iter=3000)
    print(f"{len(y)} photos, 5-fold accuracy: {cross_val_score(model, X, y, cv=5).mean():.3f}")
    model.fit(X, y)
    np.savez(
        os.path.join(MODELS, "mask_head.npz"),
        w=model.coef_[0].astype(np.float32),
        b=np.float32(model.intercept_[0]),
    )
    print("saved models/mask_head.npz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
