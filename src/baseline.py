"""
TF-IDF + Logistic Regression baseline.

This is not filler: on this specific dataset, simple TF-IDF/N-gram +
classical ML pipelines have matched or beaten deep-learning approaches
in public writeups (see README). Training this alongside the CNN gives
you an honest sense of whether the CNN is actually adding value, since
multi-class log loss on ~330 held-out rows is noisy enough that a
single number from one model tells you very little on its own.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    multi_class="multinomial",
                ),
            ),
        ]
    )


def train_and_evaluate_baseline(X_train, y_train, X_val, y_val, save_path: str):
    pipeline = build_baseline()
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)
    val_log_loss = log_loss(y_val, val_probs, labels=sorted(pipeline.classes_))

    joblib.dump(pipeline, save_path)
    print(f"Baseline validation log loss: {val_log_loss:.4f}")
    print(f"Saved baseline model to {save_path}")

    return pipeline, val_log_loss
