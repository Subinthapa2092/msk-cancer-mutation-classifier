"""
Trains the color-histogram + Logistic Regression baseline (always -- it
only needs scikit-learn) and, if TensorFlow is importable, the CNN.

Run from the project root:
    python -m src.train
    python -m src.train --model_type transfer --backbone mobilenet_v2
    python -m src.train --skip_baseline --epochs 15
"""

import argparse
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import config
from src.baseline import train_and_evaluate_baseline
from src.data_loader import load_labels
from src.preprocessing import compute_class_weights, make_splits

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def parse_args():
    parser = argparse.ArgumentParser(description="Train the histopathology classifiers")
    parser.add_argument("--labels_path", default=config.TRAIN_LABELS_PATH)
    parser.add_argument("--images_dir", default=config.TRAIN_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--model_type", choices=["scratch", "transfer"], default="scratch",
        help="CNN architecture: a from-scratch VGG-style CNN, or a pretrained backbone.",
    )
    parser.add_argument(
        "--backbone", default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50"],
        help="Only used when --model_type transfer.",
    )
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_cnn", action="store_true")
    return parser.parse_args()


def plot_training_curves(history, save_path):
    metrics = [m for m in ("loss", "auc", "accuracy") if m in history.history]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], label=f"train {metric}")
        ax.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        ax.set_title(metric)
        ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    args = parse_args()

    print(f"Loading labels from {args.labels_path} and resolving image paths under {args.images_dir}...")
    df = load_labels(args.labels_path, args.images_dir)
    print(f"Loaded {len(df)} labeled images. Class balance:\n{df['label'].value_counts()}")

    train_df, val_df, test_df = make_splits(
        df, config.VAL_SIZE, config.TEST_SIZE, config.RANDOM_STATE, label_col="label"
    )
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    y_train, y_val, y_test = (
        train_df["label"].to_numpy(),
        val_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
    )
    class_weights = compute_class_weights(y_train)
    print(f"Class weights (balanced): {class_weights}")

    # --- Baseline: always runs, needs only scikit-learn -----------------------
    if not args.skip_baseline:
        print("\nTraining color-histogram + Logistic Regression baseline...")
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        train_and_evaluate_baseline(
            train_df["filepath"], y_train, val_df["filepath"], y_val, config.BASELINE_MODEL_PATH
        )

    # --- CNN: needs TensorFlow -------------------------------------------------
    if args.skip_cnn:
        print("\n--skip_cnn passed; not training the CNN.")
        return

    if not TENSORFLOW_AVAILABLE:
        print(
            "\nTensorFlow is not installed in this environment, so the CNN was "
            "NOT trained. Install it (`pip install -r requirements.txt`) and "
            "rerun `python -m src.train` to train the CNN -- the baseline "
            "above still ran and was saved."
        )
        return

    from tensorflow import keras

    from src.dataset import build_tf_dataset
    from src.model import build_cnn_from_scratch, build_transfer_model

    print(f"\nBuilding {args.model_type} CNN...")
    if args.model_type == "scratch":
        model = build_cnn_from_scratch(
            image_size=config.IMAGE_SIZE, channels=config.IMAGE_CHANNELS,
            learning_rate=config.LEARNING_RATE,
        )
    else:
        model = build_transfer_model(
            backbone=args.backbone, image_size=config.IMAGE_SIZE,
            channels=config.IMAGE_CHANNELS,
        )
    model.summary()

    train_ds = build_tf_dataset(
        train_df["filepath"], y_train, config.IMAGE_SIZE, args.batch_size, shuffle=True, augment=True
    )
    val_ds = build_tf_dataset(val_df["filepath"], y_val, config.IMAGE_SIZE, args.batch_size)
    test_ds = build_tf_dataset(test_df["filepath"], y_test, config.IMAGE_SIZE, args.batch_size)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("\nTraining CNN...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weights, callbacks=callbacks,
    )

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.CNN_MODEL_PATH)
    print(f"Saved trained CNN to {config.CNN_MODEL_PATH}")

    plot_training_curves(history, config.TRAINING_CURVE_PATH)

    test_probs = model.predict(test_ds).ravel()
    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = np.mean((test_probs > 0.5).astype(int) == y_test)
    print(f"\nFinal CNN test ROC-AUC: {test_auc:.4f}")
    print(f"Final CNN test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
