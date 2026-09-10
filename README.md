# Histopathologic Cancer Detection CNN

**Built by Subin Thapa**

A TensorFlow/Keras deep learning project for binary classification of 96×96 RGB histopathology tissue patches. The task is to determine whether the center 32×32px region contains tumor tissue.

The project includes a from-scratch CNN trained and evaluated on real Kaggle data, a color histogram + Logistic Regression baseline, and an implemented transfer learning option using MobileNetV2, EfficientNetB0, or ResNet50.

## Results

The models were evaluated on a real test set containing **33,004 images**.

| Metric | Color Histogram + Logistic Regression | CNN |
|---|---:|---:|
| Validation ROC AUC | 0.8962 | 0.9828 |
| Test ROC AUC | 0.8937 | **0.9830** |
| Test Accuracy | 82.10% | **94.39%** |

### CNN Test Performance

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Normal | 0.95 | 0.96 | 0.95 | 19,637 |
| Tumor | 0.94 | 0.92 | 0.93 | 13,367 |

The CNN achieved a **0.9830 test ROC AUC** and **94.39% test accuracy** on the full 33,004-image test set.

## Dataset

This project uses the [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview) dataset.

The dataset contains **220,025 labeled 96×96 RGB tissue patches**. The classification target is based on whether the center 32×32px region contains tumor tissue.

The project uses a stratified split of:

```
154,017 training images
33,004 validation images
33,004 test images
```

## Model

The reported results come from a CNN built from scratch:

```
Input 96×96×3
    ↓
Rescaling
    ↓
Conv2D + BatchNorm
Conv2D + BatchNorm
MaxPooling
    ↓
Conv2D + BatchNorm
Conv2D + BatchNorm
MaxPooling
    ↓
Conv2D + BatchNorm
Conv2D + BatchNorm
MaxPooling
    ↓
Conv2D + BatchNorm
Conv2D + BatchNorm
MaxPooling
    ↓
GlobalAveragePooling2D
    ↓
Dropout 0.5
    ↓
Dense 256
    ↓
Dropout 0.3
    ↓
Sigmoid Output
```

The convolutional blocks use:

```
32 → 64 → 128 → 256 filters
```

The model uses binary cross-entropy loss, Adam optimization, ROC AUC monitoring, class-balanced weighting, early stopping, and learning rate reduction on plateau.

## Training

Training was performed on CPU only.

Because of the training time required on CPU, the CNN used a **20,000-image random subsample** of the 154,017-image training set, with **3,000 images used for validation**.

The test set remained completely untouched, and the final evaluation was performed on the full **33,004-image test set**.

Training used:

```
EarlyStopping
monitor = val_auc
patience = 5
restore_best_weights = True
```

Training stopped at epoch 14, with the best validation performance at epoch 9. The saved model contains the restored best-epoch weights.

Training on the complete training set with GPU acceleration is a natural next step for improving the model further.

## Data Pipeline

Images are loaded lazily using `tf.data` rather than loading the complete dataset into memory.

Because the dataset contains TIFF images, the pipeline uses PIL inside `tf.py_function` for image decoding.

Training augmentation includes:

```
Horizontal flips
Vertical flips
Brightness jitter
```

Augmentation is applied only to the training data.

Class-balanced weights are also used:

```python
{
    0: 0.84,
    1: 1.23
}
```

## Baseline

A color histogram + Logistic Regression model is included as a simple baseline.

The baseline extracts color information from the center 32×32px region, matching the competition's labeling rule.

Its purpose is to provide a simple reference point against which the CNN can be evaluated.

```
Image
    ↓
Center 32×32px region
    ↓
Color Histogram Features
    ↓
Logistic Regression
    ↓
Prediction
```

The baseline achieved:

```
Test ROC AUC: 0.8937
Test Accuracy: 82.10%
```

The CNN achieved:

```
Test ROC AUC: 0.9830
Test Accuracy: 94.39%
```

## Transfer Learning

The project also includes a transfer learning implementation with:

```
MobileNetV2
EfficientNetB0
ResNet50
```

These models use a pretrained backbone followed by a classification head:

```
Pretrained Backbone
    ↓
GlobalAveragePooling2D
    ↓
Dense Layer
    ↓
Sigmoid Output
```

The transfer learning implementation is available in the project but was not used for the reported results above.

## Project Structure

```
histopathologic-cancer-cnn/
│
├── data/
│
├── models/
│   ├── color_hist_logreg_baseline.joblib
│   └── histopath_cnn.keras
│
├── notebooks/
│   └── Histopathologic_Cancer_Detection_CNN.ipynb
│
├── outputs/
│   ├── class_distribution_real.png
│   ├── confusion_matrix_baseline_real.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── sample_patches_real.png
│   └── training_curves.png
│
├── src/
│   ├── __init__.py
│   ├── baseline.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── .gitignore
├── config.py
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS or Linux

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Download the Dataset

Install and configure the Kaggle CLI, then download the competition data:

```bash
kaggle competitions download -c histopathologic-cancer-detection
```

Extract the dataset into the `data` directory:

```bash
unzip histopathologic-cancer-detection.zip -d data
```

The expected structure is:

```
data/
├── train_labels.csv
├── train/
│   ├── <image_id>.tif
│   └── ...
└── test/
    ├── <image_id>.tif
    └── ...
```

You must accept the Kaggle competition rules before downloading the dataset.

## Run Training

Train the baseline and from-scratch CNN:

```bash
python -m src.train
```

Run transfer learning with MobileNetV2:

```bash
python -m src.train --model_type transfer --backbone mobilenet_v2
```

EfficientNetB0:

```bash
python -m src.train --model_type transfer --backbone efficientnet_b0
```

ResNet50:

```bash
python -m src.train --model_type transfer --backbone resnet50
```

## Evaluate

Run evaluation after training:

```bash
python -m src.evaluate
```

The evaluation pipeline produces ROC AUC, accuracy, a ROC curve, and a confusion matrix.

Generated results are saved in:

```
outputs/
```

## Single Image Prediction

Predict a single tissue patch:

```bash
python -m src.predict --image_path data/train/<image_id>.tif
```

## Docker

The project includes Docker support for a consistent environment.

Build and run:

```bash
docker compose up --build
```

Run evaluation:

```bash
docker compose run histopathologic-cancer-cnn python -m src.evaluate
```

Run prediction:

```bash
docker compose run histopathologic-cancer-cnn python -m src.predict --image_path data/train/<image_id>.tif
```

The `data`, `models`, and `outputs` directories are mounted as volumes so that datasets, trained models, and generated results remain available on the host machine.

Docker does not provide GPU acceleration for this Windows setup.

## Sample Outputs

The repository includes real sample patches and model evaluation visualizations:

```
outputs/sample_patches_real.png
outputs/class_distribution_real.png
outputs/training_curves.png
outputs/roc_curve.png
outputs/confusion_matrix.png
outputs/confusion_matrix_baseline_real.png
```

These provide visual evidence of the dataset, training behavior, model performance, and classification errors.

## Reproducibility

The project is structured around a consistent data and training pipeline with:

```
Centralized configuration
Stratified data splitting
Lazy data loading
Controlled augmentation
Class-balanced training
Saved model weights
Docker environment
requirements.txt
```

Exact neural network results can still vary between runs because of hardware, TensorFlow behavior, parallel processing, and other sources of nondeterminism.

## Next Steps

The main areas for further improvement are:

```
Train on the full 154,017-image training set
Use GPU acceleration
Benchmark transfer learning models
Fine-tune pretrained backbones
Try test-time augmentation
Experiment with model ensembling
Add Grad-CAM visualizations
Add Kaggle submission generation
Create a FastAPI prediction service
```

## Disclaimer

This project is intended for **learning and research purposes**.

It is **not a clinical diagnostic tool**, and its predictions should not be used for medical diagnosis or treatment decisions.

Real clinical deployment would require appropriate clinical validation, representative data, regulatory review, and clinician involvement.

## License

MIT License

## Author

**Subin Thapa**

Built with TensorFlow and Keras for learning and research in deep learning and medical image classification.
