"""
Loads and merges the two raw MSK competition files into one dataframe:
one row per mutation, with Gene, Variation, the clinical-literature Text,
and the Class label (1-9).

training_variants is a normal CSV: ID,Gene,Variation,Class
training_text is pipe-delimited: ID||Text  (one very long text field per row)
"""

import re

import pandas as pd


def _clean_text(text: str) -> str:
    """Light normalization. Keep it light on purpose: this is scientific
    text where punctuation, gene names, and casing can carry meaning, so
    we avoid aggressive stemming/stopword removal that a bag-of-words
    pipeline might use."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_raw(variants_path: str, text_path: str) -> pd.DataFrame:
    """Loads and merges one variants/text file pair into a single dataframe."""
    variants = pd.read_csv(variants_path)

    text_df = pd.read_csv(
        text_path,
        sep=r"\|\|",
        header=None,
        skiprows=1,
        names=["ID", "Text"],
        engine="python",
    )

    merged = variants.merge(text_df, how="left", on="ID")
    merged["Text"] = merged["Text"].apply(_clean_text)

    # A handful of rows have no matching text; drop them rather than
    # silently training on empty strings.
    missing = merged["Text"].str.len() == 0
    if missing.any():
        print(f"Dropping {missing.sum()} row(s) with no matching text.")
        merged = merged[~missing].reset_index(drop=True)

    return merged


def load_training_data(variants_path: str, text_path: str) -> pd.DataFrame:
    """Loads training_variants + training_text, with Class present.

    Also flags exact-duplicate Text values (a known issue in this dataset:
    the same literature excerpt sometimes backs multiple Gene/Variation
    rows with different labels) so callers can decide how to handle them.
    """
    df = load_raw(variants_path, text_path)

    dup_mask = df.duplicated(subset=["Text"], keep=False)
    if dup_mask.any():
        print(
            f"Note: {dup_mask.sum()} row(s) share Text with at least one "
            "other row (duplicate literature excerpts backing different "
            "Gene/Variation entries). Not dropped automatically -- see "
            "README for why."
        )

    return df


def load_test_data(variants_path: str, text_path: str) -> pd.DataFrame:
    """Loads the Kaggle test_variants/test_text pair (no Class column)."""
    return load_raw(variants_path, text_path)
