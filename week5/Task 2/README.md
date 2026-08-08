# CNN Image Classifier — MNIST Handwritten Digits

A convolutional neural network built with PyTorch to classify handwritten
digit images (0–9) from the MNIST dataset, using alternating convolutional
and max-pooling layers followed by dense layers.

> **Note on dataset choice:** The task allowed either CIFAR-10 or MNIST.
> MNIST was used here because the CIFAR-10 host was unreachable from this
> environment's network; MNIST was downloaded from a reliable GitHub mirror
> instead (see `src/dataset.py`). The same architecture pattern (conv →
> pool → dense, with Dropout/BatchNorm regularization) applies to either
> dataset — only the input channels (1 vs. 3) and image size (28×28 vs.
> 32×32) would need to change.

## CNN Architecture

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| Input | — | 1×28×28 | Grayscale image |
| Block 1 | Conv(32)+BN+ReLU ×2 → MaxPool → Dropout | 32×14×14 | 3×3 kernels, padding=1 |
| Block 2 | Conv(64)+BN+ReLU ×2 → MaxPool → Dropout | 64×7×7 | 3×3 kernels, padding=1 |
| Flatten | — | 3136 | |
| Dense | Linear(128)+BN+ReLU+Dropout | 128 | |
| Output | Linear(10) | 10 | Raw logits (softmax inside `CrossEntropyLoss`) |

- **Weight init:** Kaiming/He (matched to ReLU)
- **Regularization:** Dropout (0.15 in conv blocks, 0.3 in dense layer), BatchNorm, L2 weight decay, early stopping, data augmentation

## Preprocessing & Augmentation

- **Normalization:** per-channel mean/std standardization (MNIST global mean=0.1307, std=0.3081)
- **Training-only augmentation:** random rotation (±10°), random translation (±10%)
- **Split:** 90% of the training pool used for training, 10% held out for validation; separate untouched test set

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch size | 64 |
| Max epochs | 20 |
| Learning rate | 0.001 (Adam), halved on validation plateau |
| Weight decay | 1e-4 |
| Early stopping patience | 6 epochs |
| Training samples used | 18,000 (of 60,000 available) |
| Test samples used | 5,000 (of 10,000 available) |
| Random seed | 42 |

*(Subset sizes were chosen to keep training tractable on CPU-only hardware within a reasonable time; set `TRAIN_SUBSET = None` / `TEST_SUBSET = None` in `src/train.py` to use the full dataset given more compute.)*

## Results

| Metric | Value |
|---|---|
| Epochs run | 20 (no early stop triggered) |
| Best validation loss | 0.0242 |
| **Final test loss** | **0.0143** |
| **Final test accuracy** | **99.4%** |

Per-class precision/recall/F1 all sit at ~0.99 (see [`docs/classification_report.txt`](docs/classification_report.txt)). The [confusion matrix](docs/confusion_matrix.png) shows a strongly dominant diagonal with only a handful of misclassifications, concentrated among visually similar digit pairs (e.g. 4/9, 7/2). Training/validation loss and accuracy curves are in [`docs/loss_accuracy_curves.png`](docs/loss_accuracy_curves.png) — validation accuracy tracks slightly above training accuracy throughout, indicating the dropout + augmentation regularization is working and the model isn't overfitting.

Full write-up: [`docs/evaluation_report.md`](docs/evaluation_report.md).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train and evaluate the model (downloads MNIST if needed, trains, evaluates on test set, saves everything below):

```bash
python src/train.py
```

Outputs:
- `models/saved_cnn_model.pth` — trained model weights
- `docs/training_metrics.json` — architecture config + full per-epoch history + confusion matrix data
- `docs/loss_accuracy_curves.png` — training/validation loss & accuracy plots
- `docs/confusion_matrix.png` — confusion matrix heatmap on the test set
- `docs/classification_report.txt` — per-class precision/recall/F1

## Project Structure

```
README.md
requirements.txt
src/
  dataset.py    # data pipeline: download, normalize, augment, split
  cnn_model.py  # CNN architecture definition
  train.py      # training loop, evaluation, plots, metrics
docs/
  evaluation_report.md     # written explanation of design choices & results
  training_metrics.json    # generated after running train.py
  loss_accuracy_curves.png
  confusion_matrix.png
  classification_report.txt
models/
  saved_cnn_model.pth      # trained weights, generated after running train.py
```

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images (28×28) of handwritten digits 0–9, 60,000 for training and 10,000 for testing. Downloaded automatically by `src/dataset.py` from a GitHub mirror (https://github.com/fgnt/mnist) on first run.
