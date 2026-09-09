"""
Loads the saved CNN, re-creates the same train/val/test split used during
training (same random_state, so the test rows line up), and reports
multi-class log loss (the actual competition metric), a confusion matrix,
and a classification report.

Run from the project root:
    python -m src.evaluate
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from tensorflow import keras

import config
from src.data_loader import load_training_data
from src.preprocessing import build_feature_text, make_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN")
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    parser.add_argument("--variants_path", default=config.TRAIN_VARIANTS_PATH)
    parser.add_argument("--text_path", default=config.TRAIN_TEXT_PATH)
    return parser.parse_args()


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def main():
    args = parse_args()

    model = keras.models.load_model(args.model_path)

    df = load_training_data(args.variants_path, args.text_path)
    df["feature_text"] = build_feature_text(df)

    # Same split, same random_state -> the same test rows as training time.
    _, _, test_df = make_splits(
        df, val_size=config.VAL_SIZE, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    y_test = (test_df["Class"] - 1).to_numpy()

    probs = model.predict(test_df["feature_text"].to_numpy())
    y_pred = np.argmax(probs, axis=1)

    test_log_loss = log_loss(y_test, probs, labels=list(range(config.NUM_CLASSES)))
    test_accuracy = np.mean(y_pred == y_test)

    print(f"Test log loss: {test_log_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    class_labels = [str(i + 1) for i in range(config.NUM_CLASSES)]

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(config.NUM_CLASSES)))
    print(cm)
    plot_confusion_matrix(cm, config.CONFUSION_MATRIX_PATH)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, labels=list(range(config.NUM_CLASSES)), target_names=class_labels
        )
    )


if __name__ == "__main__":
    main()
