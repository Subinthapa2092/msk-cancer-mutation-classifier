# models/

This folder contains the two trained models produced by this project.

## `histopath_cnn.keras` — real, trained on the actual Kaggle data

Trained on a 20,000-image subsample of the real 154,017-image training split (3,000-image validation subsample), CPU-only. Early stopping (`patience=5`, monitoring `val_auc`) stopped training at epoch 14; these are the restored best-epoch (epoch 9) weights, not the final epoch's.

Real results, evaluated on the full, untouched 33,004-image test set:

- Test ROC-AUC: 0.9830
- Test accuracy: 0.9439
- Tumor class: precision 0.94, recall 0.92
- Normal class: precision 0.95, recall 0.96

Load it directly:

```python
from tensorflow import keras
model = keras.models.load_model("models/histopath_cnn.keras")
```

Not trained on the full training set or on a GPU — see the root `README.md`'s "Next Steps" section for what would likely push this score higher still.

## `color_hist_logreg_baseline.joblib` — real, trained on the actual Kaggle data

The color-histogram + Logistic Regression baseline, trained on the full 154,017-image real training split (this one's fast enough that no subsampling was needed).

Real results:

- Val ROC-AUC: 0.8962
- Test ROC-AUC: 0.8937
- Test accuracy: 0.8210

Load it directly:

```python
import joblib
pipeline = joblib.load("models/color_hist_logreg_baseline.joblib")
```

As expected, well behind the CNN — tumor detection in tissue patches depends on spatial/textural structure that a color histogram doesn't capture. Kept as the reference point the CNN needed to beat.