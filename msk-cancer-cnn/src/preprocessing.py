"""
Turns the merged dataframe into model-ready features:
  - a single text field per row (Gene + Variation + clinical Text), since
    the gene/variation identity is informative and cheap to fold into the
    token stream the CNN reads
  - stratified train/val/test splits (important here: 9 classes, ~3300
    rows total, and some classes have very few examples)
  - class weights, to fight the class imbalance called out in the
    competition write-ups instead of pretending it isn't there
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def build_feature_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["Gene"].fillna("") + " " + df["Variation"].fillna("") + " " + df["Text"].fillna("")
    )


def make_splits(df: pd.DataFrame, val_size: float, test_size: float, random_state: int):
    """Two-stage stratified split: train / val / test, each preserving the
    class distribution as closely as sklearn's stratify allows."""
    labels = df["Class"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["Class"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights(labels: np.ndarray) -> dict:
    """labels are expected to be 0-indexed (Class - 1)."""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))
