# MLP Binary Classifier — Breast Cancer Wisconsin (Diagnostic) Dataset

A from-scratch feedforward neural network (multi-layer perceptron) built with
PyTorch to classify tumors as malignant or benign from 30 numeric diagnostic
features.

## Neural Network Architecture

| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | — | 30 (features) | — |
| Hidden 1 | Linear + BatchNorm | 64 | ReLU (+ Dropout 0.2) |
| Hidden 2 | Linear + BatchNorm | 32 | ReLU (+ Dropout 0.2) |
| Hidden 3 | Linear + BatchNorm | 16 | ReLU (+ Dropout 0.2) |
| Output | Linear | 1 | Sigmoid (applied inside `BCEWithLogitsLoss`) |

- **Weight init:** Kaiming/He normal (matched to ReLU)
- **Loss:** Binary Cross-Entropy with logits (`nn.BCEWithLogitsLoss`)
- **Optimizer:** Adam (`lr=1e-3`, `weight_decay=1e-4`)
- **LR schedule:** `ReduceLROnPlateau` on validation loss
- **Regularization:** Dropout (0.2), BatchNorm, L2 weight decay, early stopping

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max epochs | 100 |
| Learning rate | 0.001 |
| Early stopping patience | 15 epochs |
| Train / Val / Test split | 70% / 15% / 15% (stratified) |
| Random seed | 42 |

## Training Results

The model trains until early stopping and restores the best-validation-loss
checkpoint before final evaluation.

| Metric | Value |
|---|---|
| Epochs run | 100 |
| Best validation loss | 0.0466 |
| **Final test loss** | **0.0895** |
| **Final test accuracy** | **96.4%** |

Full per-epoch loss/accuracy history is saved to
[`docs/training_metrics.json`](docs/training_metrics.json), and a fuller
write-up is in [`docs/architecture_report.md`](docs/architecture_report.md).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model (loads data, trains, evaluates on test set, saves weights + metrics):

```bash
python src/train.py
```

Outputs:
- `trained_model.pt` — trained model weights (repo root)
- `docs/training_metrics.json` — architecture config + full training history

## Project Structure

```
README.md
requirements.txt
src/
  model.py    # MLP architecture definition
  train.py    # data loading, training loop, evaluation
docs/
  architecture_report.md   # written explanation of design choices
  training_metrics.json    # generated after running train.py
```

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
— 569 samples, 30 real-valued features computed from digitized images of
fine needle aspirate (FNA) of breast masses, binary target (malignant /
benign). Loaded via `sklearn.datasets.load_breast_cancer()`, so no external
download is required.
