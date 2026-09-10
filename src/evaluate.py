"""
Evaluates the trained CNN on the held-out test split (same random_state
as training, so the same rows), reporting ROC-AUC (the competition
metric), a ROC curve, a confusion matrix, and a classification report.

Run from the project root:
    python -m src.evaluate
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

import config
from src.data_loader import load_labels
from src.dataset import build_tf_dataset
from src.preprocessing import make_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN")
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    parser.add_argument("--labels_path", default=config.TRAIN_LABELS_PATH)
    parser.add_argument("--images_dir", default=config.TRAIN_DIR)
    return parser.parse_args()


def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"CNN (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved ROC curve to {save_path}")


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Tumor"], yticklabels=["Normal", "Tumor"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def main():
    args = parse_args()

    model = keras.models.load_model(args.model_path)

    df = load_labels(args.labels_path, args.images_dir)
    _, _, test_df = make_splits(
        df, config.VAL_SIZE, config.TEST_SIZE, config.RANDOM_STATE, label_col="label"
    )
    y_test = test_df["label"].to_numpy()

    test_ds = build_tf_dataset(test_df["filepath"], y_test, config.IMAGE_SIZE, config.BATCH_SIZE)
    probs = model.predict(test_ds).ravel()
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    print(f"Test ROC-AUC: {auc:.4f}")

    plot_roc_curve(y_test, probs, config.ROC_CURVE_PATH)

    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, config.CONFUSION_MATRIX_PATH)

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Normal", "Tumor"]))


if __name__ == "__main__":
    main()
