"""
Predicts tumor/no-tumor for a single patch image using the saved CNN.

Run from the project root:
    python -m src.predict --image_path path/to/patch.tif
"""

import argparse

import numpy as np
from tensorflow import keras

import config
from src.preprocessing import load_image_as_array


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tumor presence for one patch image")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for the positive (tumor) class.",
    )
    return parser.parse_args()


def predict_image(image_path: str, model, image_size: int, threshold: float) -> dict:
    image = load_image_as_array(image_path, size=image_size).astype(np.float32)
    image = np.expand_dims(image, axis=0)  # add batch dim
    prob = float(model.predict(image, verbose=0)[0][0])
    return {
        "tumor_probability": prob,
        "predicted_label": "Tumor" if prob >= threshold else "Normal",
    }


def main():
    args = parse_args()
    model = keras.models.load_model(args.model_path)
    result = predict_image(args.image_path, model, config.IMAGE_SIZE, args.threshold)

    print(f"Predicted: {result['predicted_label']}")
    print(f"Tumor probability: {result['tumor_probability']:.4f}")


if __name__ == "__main__":
    main()
