"""
Central configuration for the MSK "Redefining Cancer Treatment" project.
Edit the paths below to match where you downloaded the Kaggle competition
files (msk-redefining-cancer-treatment).
"""

import os

# --- Dataset paths -------------------------------------------------------
# Expected files (from the Kaggle competition, unzipped into data/):
#   data/training_variants          (ID, Gene, Variation, Class)
#   data/training_text              (ID||Text, pipe-delimited)
#   data/test_variants   (optional, for the Kaggle leaderboard submission)
#   data/test_text        (optional, for the Kaggle leaderboard submission)
DATA_DIR = "data"
TRAIN_VARIANTS_PATH = os.path.join(DATA_DIR, "training_variants")
TRAIN_TEXT_PATH = os.path.join(DATA_DIR, "training_text")
TEST_VARIANTS_PATH = os.path.join(DATA_DIR, "test_variants")
TEST_TEXT_PATH = os.path.join(DATA_DIR, "test_text")

NUM_CLASSES = 9  # classes are labeled 1-9 in the raw data

# --- Split ----------------------------------------------------------------
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# --- Text / tokenization ---------------------------------------------------
MAX_VOCAB_SIZE = 30000
MAX_SEQUENCE_LENGTH = 1500   # truncate/pad clinical-text excerpts to this many tokens
EMBEDDING_DIM = 128

# --- Training hyperparameters ----------------------------------------------
BATCH_SIZE = 16              # dataset is small (~3300 rows); keep batches modest
EPOCHS = 30
LEARNING_RATE = 1e-3

# --- Output paths -----------------------------------------------------------
MODEL_DIR = "models"
# TextVectorization is a layer inside the CNN model, so its vocabulary is
# saved and restored automatically with the model -- no separate tokenizer
# file needed.
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "text_cnn.keras")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg_baseline.joblib")

OUTPUTS_DIR = "outputs"
TRAINING_CURVE_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
