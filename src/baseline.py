"""
A classical, non-deep-learning baseline for the same task: color-histogram
+ basic per-channel statistics features, fed into Logistic Regression.

Why this exists: it trains in seconds on CPU with no GPU/TensorFlow
required, so it's a real, runnable sanity check for the data pipeline
even in environments where the CNN itself can't be trained. It is NOT
expected to be competitive with the CNN on this task -- tumor detection
in tissue patches depends on spatial/textural structure that a
histogram genuinely throws away -- but a working baseline beats no
baseline, and its AUC gives you a floor to compare the CNN against.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_image_as_array


def extract_color_features(filepath: str, bins: int = 16) -> np.ndarray:
    img = load_image_as_array(filepath).astype(np.float32) / 255.0

    features = []
    for channel in range(3):
        channel_pixels = img[:, :, channel].ravel()
        hist, _ = np.histogram(channel_pixels, bins=bins, range=(0.0, 1.0))
        hist = hist / hist.sum()  # normalize so it doesn't just encode patch size
        features.extend(hist.tolist())
        features.append(float(channel_pixels.mean()))
        features.append(float(channel_pixels.std()))

    # The label only depends on the CENTER 32x32 region, per the
    # competition's own labeling rule -- so features from that region
    # specifically are worth including alongside the whole-patch ones.
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    center = img[cy - 16 : cy + 16, cx - 16 : cx + 16, :]
    for channel in range(3):
        center_pixels = center[:, :, channel].ravel()
        features.append(float(center_pixels.mean()))
        features.append(float(center_pixels.std()))

    return np.array(features, dtype=np.float32)


def build_feature_matrix(filepaths) -> np.ndarray:
    return np.stack([extract_color_features(fp) for fp in filepaths])


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]
    )


def train_and_evaluate_baseline(train_filepaths, y_train, val_filepaths, y_val, save_path: str):
    print("Extracting color-histogram features for train/val (this reads every image once)...")
    X_train = build_feature_matrix(train_filepaths)
    X_val = build_feature_matrix(val_filepaths)

    pipeline = build_baseline()
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    joblib.dump(pipeline, save_path)
    print(f"Baseline validation ROC-AUC: {val_auc:.4f}")
    print(f"Saved baseline model to {save_path}")

    return pipeline, val_auc
