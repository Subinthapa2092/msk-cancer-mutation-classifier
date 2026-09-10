"""
Loads train_labels.csv and resolves each row to its image file path under
data/train/. Deliberately does NOT load all 130k+ images into memory --
that's ~4GB+ of RGB arrays for the real dataset. Downstream code reads
images lazily (tf.data / a Keras generator) at train time.
"""

import os

import pandas as pd


def load_labels(labels_path: str, images_dir: str, image_ext: str = ".tif") -> pd.DataFrame:
    """Returns a dataframe with columns: id, label, filepath.

    Rows whose image file is missing from images_dir are dropped (with a
    warning) rather than silently kept -- a truncated/partial download is
    a common real-world failure mode with a dataset this size.
    """
    labels = pd.read_csv(labels_path)
    if not {"id", "label"}.issubset(labels.columns):
        raise ValueError(f"Expected columns 'id' and 'label' in {labels_path}")

    labels["filepath"] = labels["id"].apply(lambda i: os.path.join(images_dir, f"{i}{image_ext}"))

    exists_mask = labels["filepath"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing:
        print(
            f"Warning: {missing} row(s) in {labels_path} have no matching "
            f"image file under {images_dir} -- dropping them. This usually "
            "means an incomplete download/unzip."
        )
        labels = labels[exists_mask].reset_index(drop=True)

    return labels


def load_test_filepaths(images_dir: str, image_ext: str = ".tif") -> pd.DataFrame:
    """For the unlabeled Kaggle test set: one row per image file found."""
    ids = [
        f[: -len(image_ext)]
        for f in os.listdir(images_dir)
        if f.endswith(image_ext)
    ]
    return pd.DataFrame(
        {"id": ids, "filepath": [os.path.join(images_dir, f"{i}{image_ext}") for i in ids]}
    )
