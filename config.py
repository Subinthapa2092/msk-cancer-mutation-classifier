"""
Central configuration for the Kaggle "Histopathologic Cancer Detection"
project (PatchCamelyon-derived). Edit the paths below to match where you
downloaded the competition files.
"""

import os
## 
# --- Dataset paths ---------------------------------------------------------
# Expected layout (from `kaggle competitions download -c
# histopathologic-cancer-detection`, unzipped into data/):
#   data/train_labels.csv     (id, label)  -- label is 1 if the center
#                                              32x32px region of the patch
#                                              contains tumor tissue
#   data/train/<id>.tif        96x96x3 RGB patches, ~130k+ files
#   data/test/<id>.tif         unlabeled patches for the Kaggle leaderboard
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")

IMAGE_SIZE = 96          # native patch size; do not upsample needlessly
IMAGE_CHANNELS = 3
NUM_CLASSES = 2           # binary: tumor present / not present

# --- Split -------------------------------------------------------------------
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# --- Training hyperparameters -------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

# --- Output paths ---------------------------------------------------------------
MODEL_DIR = "models"
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "histopath_cnn.keras")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "color_hist_logreg_baseline.joblib")

OUTPUTS_DIR = "outputs"
TRAINING_CURVE_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")
ROC_CURVE_PATH = os.path.join(OUTPUTS_DIR, "roc_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
