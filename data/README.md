# data/

This folder is intentionally empty in the GitHub repository.

The dataset (220,025 labeled 96×96 RGB tissue patches, ~7GB) is excluded via `.gitignore` because of its size — it is not committed to version control.

## How to populate this folder

See the "Download the Dataset" section in the root `README.md`, or run:

```bash
kaggle competitions download -c histopathologic-cancer-detection
unzip histopathologic-cancer-detection.zip -d data
```

After extraction, this folder should contain:

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

You must accept the Kaggle competition rules before downloading.