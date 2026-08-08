# Evaluation Report

## 1. Problem and Dataset

This project trains a CNN to classify handwritten digit images (0–9) from
**MNIST** — 70,000 grayscale 28×28 images, 60,000 for training and 10,000
for testing.

**Dataset note:** The task brief allowed CIFAR-10 or MNIST. CIFAR-10's host
was not reachable from this project's network environment, so MNIST was
used instead, sourced from a GitHub mirror (`fgnt/mnist`) rather than the
original yann.lecun.com host, which is frequently slow or unreachable. The
preprocessing/augmentation/architecture pattern below generalizes directly
to CIFAR-10 — only the input channel count (1 → 3) and spatial size
(28×28 → 32×32) would need to change.

To keep training time reasonable on CPU-only hardware, the model was
trained on an 18,000-image subset (of the 60,000 available) and evaluated
on a 5,000-image test subset (of the 10,000 available), both drawn with a
fixed random seed for reproducibility.

## 2. Preprocessing and Augmentation

- **Normalization:** each pixel is standardized using MNIST's global
  mean (0.1307) and standard deviation (0.3081), which keeps input values
  centered near zero and speeds up convergence.
- **Data augmentation (training set only):**
  - Random rotation up to ±10°
  - Random translation up to ±10% of image size in both axes

  These simulate natural variation in handwriting (slight tilts, off-center
  digits) so the model doesn't overfit to the exact pixel positions seen in
  training. Augmentation is applied only to the training split — validation
  and test data use only normalization, so evaluation reflects real-world
  performance rather than an artificially easier augmented distribution.
- **Splits:** 90% of the training pool is used for training and 10% held
  out for validation (used for early stopping and LR scheduling); the test
  set is never touched until final evaluation.

## 3. CNN Architecture

```
Input (1×28×28)
  → [Conv3x3(32) → BatchNorm → ReLU] × 2 → MaxPool(2) → Dropout(0.15)   (→ 32×14×14)
  → [Conv3x3(64) → BatchNorm → ReLU] × 2 → MaxPool(2) → Dropout(0.15)   (→ 64×7×7)
  → Flatten (3136)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(10)  → [Softmax, applied inside CrossEntropyLoss]
```

**Design rationale:**
- **Alternating conv + pooling blocks:** each block extracts increasingly
  abstract spatial features — early layers pick up edges and strokes,
  later layers combine these into digit-level shapes — while max-pooling
  progressively shrinks the spatial resolution (28→14→7) and reduces
  parameter count and computation in later layers.
- **Two conv layers per block before pooling:** stacking convolutions
  before downsampling lets the network build a richer feature
  representation at each resolution before discarding spatial detail.
- **BatchNorm after every conv/linear layer:** stabilizes training by
  normalizing layer inputs, allowing a higher learning rate and faster
  convergence.
- **Dropout, increasing with depth (0.15 → 0.3):** randomly disables
  activations during training to prevent co-adaptation and overfitting;
  a higher rate in the dense layer (which has the most parameters and
  highest overfitting risk) than in the earlier conv blocks.
- **Kaiming/He initialization:** matched to ReLU activations, keeps
  activation variance stable through the network at the start of training.

## 4. Activation Functions

- **Hidden layers (conv and dense) — ReLU:** avoids vanishing gradients
  for positive inputs and is computationally cheap, standard choice for
  deep CNNs.
- **Output layer — Softmax (implicit):** converts the 10 output logits
  into a probability distribution over digit classes. As with the loss
  function below, this is applied inside `nn.CrossEntropyLoss` rather than
  as an explicit layer, for numerical stability (avoids computing
  `log(softmax(x))` directly).

## 5. Loss Function, Backpropagation, and Optimization

- **Loss — Cross-Entropy:** the standard loss for multi-class
  classification; it penalizes the model heavily for assigning low
  probability to the correct class, pushing predictions toward confident,
  correct probability distributions.
- **Backpropagation:** `loss.backward()` computes the gradient of the loss
  with respect to every learnable parameter via reverse-mode automatic
  differentiation, propagating error signal from the output layer back
  through the dense layer, both conv blocks, to the input.
- **Optimizer — Adam:** adapts the learning rate per-parameter using
  running estimates of gradient moments, which converges faster and is
  less sensitive to the initial learning rate than plain SGD — a good fit
  given the limited compute budget for this run.
- **Learning rate schedule:** `ReduceLROnPlateau` halves the learning rate
  when validation loss stalls for 3 epochs, allowing coarse progress early
  and fine-tuning later.
- **Weight decay (L2, 1e-4):** penalizes large weights directly in the
  Adam optimizer, an additional regularizer alongside Dropout/BatchNorm.

## 6. Training Loop and Overfitting Prevention

Each epoch: shuffle and batch the training data → forward pass → compute
cross-entropy loss → backward pass → Adam step (training only), then
evaluate the same-epoch model on the validation set with no gradient
updates. Both losses/accuracies are logged every epoch, and the
best-validation-loss checkpoint (not necessarily the last epoch) is
restored before final test evaluation — the best epoch's weights are
what actually gets saved to `models/saved_cnn_model.pth`.

**Overfitting was monitored via the train/validation gap** (see
`docs/loss_accuracy_curves.png`): validation accuracy stayed at or above
training accuracy for the entire run, and validation loss declined roughly
in step with training loss without diverging upward — evidence that the
combination of data augmentation, dropout, batch norm, and weight decay
successfully controlled overfitting even on a training subset smaller than
the full dataset.

## 7. Final Results

| Split | Loss | Accuracy |
|---|---|---|
| Best validation checkpoint | 0.0242 | 99.2–99.4% (epoch-dependent, see curve) |
| **Test (held-out, unseen)** | **0.0143** | **99.4%** |

Per-class metrics (from `docs/classification_report.txt`):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.994 | 0.998 | 0.996 |
| 1 | 0.993 | 0.995 | 0.994 |
| 2 | 0.987 | 0.996 | 0.992 |
| 3 | 0.996 | 0.998 | 0.997 |
| 4 | 0.992 | 0.998 | 0.995 |
| 5 | 0.995 | 0.995 | 0.995 |
| 6 | 0.998 | 0.992 | 0.995 |
| 7 | 0.988 | 0.990 | 0.989 |
| 8 | 1.000 | 0.987 | 0.994 |
| 9 | 0.998 | 0.990 | 0.994 |
| **Macro avg** | **0.994** | **0.994** | **0.994** |

The [confusion matrix](confusion_matrix.png) shows an overwhelmingly
dominant diagonal — nearly all 5,000 test images were classified correctly,
with the small number of errors scattered rather than concentrated,
suggesting no single class is systematically problematic for the model.

## 8. Reproducibility

All randomness (data subsetting, train/val split, weight init, batch
shuffling) is seeded (`SEED = 42`). Running `python src/train.py` from a
clean checkout reproduces the same architecture, training procedure, and
(modulo minor floating-point/hardware nondeterminism) very similar results,
regenerating all files in `docs/` and `models/`.
