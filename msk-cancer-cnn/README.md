# MSK Redefining Cancer Treatment — Clinical Text Classifier (CNN)

A 9-class classifier that reads a Gene, a Variation, and a clinical-literature
excerpt, and predicts which of 9 expert-annotated mutation classes it belongs
to — built with a 1D Convolutional Neural Network in TensorFlow/Keras, plus a
TF-IDF + Logistic Regression baseline trained alongside it for an honest
sanity check.

**Dataset:** [Kaggle — MSK Redefining Cancer Treatment](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/overview)

## Important: this is a text classification problem, not an image one

Despite the "CNN" framing, the input here is text (a table of Gene +
Variation pairs joined to a long clinical-text excerpt), not pixels. A 1D
CNN over word embeddings is used below because a CNN was requested, but it's
worth knowing upfront that on this specific dataset:

- The top public leaderboard solutions used **TF-IDF/Word2Vec features +
  gradient-boosted trees** (LightGBM/XGBoost), not deep text models.
- **BioBERT** (a transformer pretrained on biomedical text) has outperformed
  CNN/LSTM/BiLSTM baselines in published comparisons on this task.
- The dataset itself is small (~3,300 labeled rows) and imbalanced across
  9 classes, which limits how much a from-scratch embedding + CNN can learn
  compared to a pretrained transformer or a well-tuned tree ensemble.

That's why `src/baseline.py` trains a TF-IDF + Logistic Regression model in
the same run as the CNN (`python -m src.train`) — treat its log loss as the
number to beat, not as a throwaway comparison. If the CNN doesn't clearly
beat it, that's a legitimate, expected outcome for this dataset, not a sign
something is broken.

## Clone

```bash
git clone <your-repo-url>
cd msk-cancer-cnn
```

## Get the data

1. Download `training_variants` and `training_text` from the
   [competition data page](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/data)
   (Kaggle login required; you may need to accept the competition rules).
2. Place both files, unzipped, directly under `data/`:

```
data/
  training_variants
  training_text
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python -m src.train                                   # trains baseline + CNN
python -m src.evaluate                                 # log loss, confusion matrix
python -m src.predict --gene BRCA1 --variation "S1655F" --text_file excerpt.txt
```

## Run with Docker

```bash
docker compose up --build                                              # build + train
docker compose run msk-cancer-classifier python -m src.evaluate
docker compose run msk-cancer-classifier python -m src.predict --gene TP53 --variation "R175H" --text_file excerpt.txt
```
`data/`, `models/`, and `outputs/` are mounted as volumes, so results are
saved back to your machine, not lost inside the container.

## Model

```
Text(Gene+Variation+Excerpt) -> TextVectorization -> Embedding(128)
  -> [Conv1D(k=3,4,5) -> GlobalMaxPool1D] in parallel   (Kim, 2014 style)
  -> Concatenate -> Dropout(0.4) -> Dense(128) -> Dropout(0.4) -> Dense(9, softmax)
```
`sparse_categorical_crossentropy` loss (matches the competition's log-loss
metric) · `adam` optimizer · class-balanced weighting · early stopping
(patience=5) · LR reduction on plateau.

**Design choices that matter for this dataset specifically:**
- **Gene + Variation are prepended to the text** before vectorization,
  since the mutation identity itself is informative and costs nothing to
  fold into the token stream.
- **Class weights are balanced**, not left at default, because class sizes
  in this competition are known to be skewed.
- **`output_sequence_length=1500`** truncates/pads the (often very long)
  literature excerpts; this is a real information loss worth tuning if you
  have GPU budget to raise it.
- **Duplicate `Text` values are flagged, not silently dropped** — some
  rows legitimately share a literature excerpt across different
  Gene/Variation entries with different labels, per the known dataset
  quirks; blindly deduping on Text would lose real training examples.

## Results

Not filled in here on purpose — this is a template you run yourself once
you've downloaded the actual Kaggle data (I don't have network/dataset
access in this environment). After `python -m src.train` and
`python -m src.evaluate`, fill in:

| Metric               | Baseline (TF-IDF+LogReg) | CNN |
|-----------------------|--------------------------|-----|
| Val log loss          |                          |     |
| Test log loss         |                          |     |
| Test accuracy         |                          |     |

**Confusion Matrix:** saved to `outputs/confusion_matrix.png` after running
`python -m src.evaluate`.

## Next steps toward "production"

This repo covers a solid, reproducible baseline. If you want to push
further:
- Swap the from-scratch embedding for pretrained biomedical embeddings
  (e.g. BioWordVec) or fine-tune BioBERT — both have outperformed CNNs on
  this exact competition in published work.
- Add k-fold cross-validation given how small the labeled set is; a single
  train/val/test split on ~3,300 rows has real variance.
- Wrap `src/predict.py` in a small FastAPI service if you want an actual
  served endpoint rather than a CLI.

Licensed under MIT (see `LICENSE`, add your own if publishing).
