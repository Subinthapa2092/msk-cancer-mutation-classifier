"""
Splitting, class weighting, and image I/O helpers shared by the baseline
and the CNN.
"""

import warnings

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def _safe_stratified_split(frame, test_size: float, random_state: int, label_col: str):
    """train_test_split(stratify=...), falling back to an unstratified
    split (with a warning) instead of raising when a class has too few
    rows to appear on both sides of the split."""
    counts = frame[label_col].value_counts()
    min_count = counts.min()

    if min_count < 2:
        warnings.warn(
            f"Class(es) with only {min_count} row(s) present; falling back "
            "to an unstratified split for this stage."
        )
        return train_test_split(frame, test_size=test_size, random_state=random_state)

    try:
        return train_test_split(
            frame, test_size=test_size, stratify=frame[label_col], random_state=random_state
        )
    except ValueError as exc:
        warnings.warn(f"Stratified split failed ({exc}); falling back to unstratified split.")
        return train_test_split(frame, test_size=test_size, random_state=random_state)


def make_splits(df, val_size: float, test_size: float, random_state: int, label_col: str = "label"):
    """Two-stage split: train / val / test, stratified by label where the
    class counts allow it."""
    train_df, temp_df = _safe_stratified_split(
        df, test_size=(val_size + test_size), random_state=random_state, label_col=label_col
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = _safe_stratified_split(
        temp_df, test_size=relative_test_size, random_state=random_state, label_col=label_col
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights(labels: np.ndarray) -> dict:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


def load_image_as_array(filepath: str, size: int = 96) -> np.ndarray:
    """Loads one image as an HxWx3 uint8 array, resizing only if it isn't
    already the expected patch size (the real dataset's patches already
    are, so this is normally a no-op cost-wise)."""
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size))
        return np.array(img)
