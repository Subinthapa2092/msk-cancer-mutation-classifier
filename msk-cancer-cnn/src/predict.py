"""
Predicts the mutation class for a single Gene/Variation/Text example using
the saved CNN.

Run from the project root:
    python -m src.predict --gene BRCA1 --variation "S1655F" --text "..."
    python -m src.predict --text_file path/to/excerpt.txt --gene TP53 --variation "R175H"
"""

import argparse

import numpy as np
from tensorflow import keras

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Predict mutation class for one example")
    parser.add_argument("--gene", required=True, help="Gene symbol, e.g. BRCA1")
    parser.add_argument("--variation", required=True, help="Variation, e.g. S1655F")
    parser.add_argument("--text", help="Clinical literature excerpt as a raw string")
    parser.add_argument("--text_file", help="Path to a text file containing the excerpt instead")
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    return parser.parse_args()


def predict_example(gene: str, variation: str, text: str, model) -> dict:
    feature_text = f"{gene} {variation} {text}"
    probs = model.predict(np.array([feature_text]))[0]
    predicted_class = int(np.argmax(probs)) + 1  # back to 1-9 labeling
    return {
        "predicted_class": predicted_class,
        "confidence": float(probs[predicted_class - 1]),
        "all_class_probabilities": {i + 1: float(p) for i, p in enumerate(probs)},
    }


def main():
    args = parse_args()

    if not args.text and not args.text_file:
        raise SystemExit("Provide either --text or --text_file.")

    text = args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()

    model = keras.models.load_model(args.model_path)
    result = predict_example(args.gene, args.variation, text, model)

    print(f"Predicted class: {result['predicted_class']} ({result['confidence']:.2%} confidence)")
    print("Full class probabilities:")
    for class_id, prob in sorted(result["all_class_probabilities"].items()):
        print(f"  Class {class_id}: {prob:.4f}")


if __name__ == "__main__":
    main()
