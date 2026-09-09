"""
Trains the clinical-text CNN (and, alongside it, the TF-IDF + Logistic
Regression baseline) and saves both models plus a training-curve plot.

Run from the project root:
    python -m src.train
    python -m src.train --epochs 15 --batch_size 32
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
from tensorflow import keras
from tensorflow.keras import layers

import config
from src.baseline import train_and_evaluate_baseline
from src.data_loader import load_training_data
from src.model import build_text_cnn
from src.preprocessing import build_feature_text, compute_class_weights, make_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Train the clinical-text CNN")
    parser.add_argument("--variants_path", default=config.TRAIN_VARIANTS_PATH)
    parser.add_argument("--text_path", default=config.TRAIN_TEXT_PATH)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip training the TF-IDF + LogReg baseline (CNN-only run)",
    )
    return parser.parse_args()


def plot_training_curves(history, save_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    args = parse_args()

    print("Loading and merging training_variants + training_text...")
    df = load_training_data(args.variants_path, args.text_path)
    print(f"Loaded {len(df)} rows across {df['Class'].nunique()} classes.")

    df["feature_text"] = build_feature_text(df)

    train_df, val_df, test_df = make_splits(
        df, val_size=config.VAL_SIZE, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    y_train = (train_df["Class"] - 1).to_numpy()
    y_val = (val_df["Class"] - 1).to_numpy()
    y_test = (test_df["Class"] - 1).to_numpy()

    class_weights = compute_class_weights(y_train)
    print(f"Class weights (balanced): {class_weights}")

    # --- Baseline (TF-IDF + Logistic Regression) --------------------------
    if not args.skip_baseline:
        print("\nTraining TF-IDF + Logistic Regression baseline...")
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        train_and_evaluate_baseline(
            train_df["feature_text"], y_train, val_df["feature_text"], y_val, config.BASELINE_MODEL_PATH
        )

    # --- CNN ----------------------------------------------------------------
    print("\nBuilding TextVectorization layer and adapting on training text...")
    vectorize_layer = layers.TextVectorization(
        max_tokens=config.MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=config.MAX_SEQUENCE_LENGTH,
    )
    vectorize_layer.adapt(train_df["feature_text"].to_numpy())
    vocab_size = vectorize_layer.vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")

    print("Building CNN model...")
    model = build_text_cnn(
        vectorize_layer,
        vocab_size=vocab_size,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        learning_rate=config.LEARNING_RATE,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("\nTraining CNN...")
    history = model.fit(
        train_df["feature_text"].to_numpy(),
        y_train,
        validation_data=(val_df["feature_text"].to_numpy(), y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.CNN_MODEL_PATH)
    print(f"Saved trained CNN to {config.CNN_MODEL_PATH}")

    plot_training_curves(history, config.TRAINING_CURVE_PATH)

    # --- Held-out test evaluation (multi-class log loss, the comp metric) --
    test_probs = model.predict(test_df["feature_text"].to_numpy())
    test_log_loss = log_loss(y_test, test_probs, labels=list(range(config.NUM_CLASSES)))
    test_accuracy = np.mean(np.argmax(test_probs, axis=1) == y_test)

    print(f"\nFinal CNN test log loss: {test_log_loss:.4f}")
    print(f"Final CNN test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
