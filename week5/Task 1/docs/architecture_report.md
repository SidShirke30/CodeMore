# Architecture & Training Report

## 1. Problem and Dataset

This project trains a multi-layer perceptron (MLP) to perform **binary
classification** on the Breast Cancer Wisconsin (Diagnostic) dataset: given
30 numeric features derived from a digitized image of a breast mass (e.g.
radius, texture, perimeter, smoothness, concavity — each summarized as
mean, standard error, and "worst" value), predict whether the mass is
**malignant** or **benign**.

- 569 total samples, 30 continuous input features, 1 binary target.
- Split 70% train / 15% validation / 15% test, stratified by class to
  preserve the class balance in each split.
- Features are standardized (zero mean, unit variance) using a `StandardScaler`
  fit only on the training set, then applied to validation/test — this avoids
  leaking test-set statistics into training.

## 2. Network Architecture

The model (`src/model.py`) is a standard feedforward MLP:

```
Input (30) → Linear(64) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(16) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(1)  → [Sigmoid, applied inside the loss]
```

**Design rationale:**
- **Funnel-shaped hidden layers (64 → 32 → 16):** progressively compress the
  30 raw features into a more abstract, lower-dimensional representation
  before the final decision layer.
- **BatchNorm after each linear layer:** stabilizes and speeds up training by
  keeping layer inputs at a consistent scale, and provides a mild
  regularization effect.
- **Dropout (p=0.2):** randomly zeroes 20% of activations during training to
  discourage co-adaptation of neurons and reduce overfitting.
- **Kaiming/He weight initialization:** designed for ReLU activations, this
  keeps the variance of activations stable across layers at the start of
  training (unlike default/Xavier init, which assumes symmetric activations
  like tanh).

## 3. Activation Functions

- **Hidden layers — ReLU:** `f(x) = max(0, x)`. Chosen over sigmoid/tanh for
  hidden layers because it doesn't saturate for positive inputs, which keeps
  gradients from vanishing as they propagate backward through multiple
  layers, and it's computationally cheap.
- **Output layer — Sigmoid:** squashes the single output logit into a
  probability in (0, 1), appropriate for binary classification. In the code,
  this is applied *implicitly* inside `nn.BCEWithLogitsLoss` rather than as a
  separate `nn.Sigmoid()` layer, because combining the sigmoid and the
  cross-entropy loss into one operation is numerically more stable (it avoids
  computing `log(sigmoid(x))` directly, which can underflow for very
  negative/positive logits).

## 4. Loss Function

**Binary Cross-Entropy (BCE)** is the natural loss for binary classification
with a probabilistic output:

```
L = -[y·log(p) + (1-y)·log(1-p)]
```

where `y` is the true label (0 or 1) and `p` is the predicted probability.
It penalizes confident-and-wrong predictions much more heavily than
confident-and-correct ones, which pushes the model toward well-calibrated
probabilities rather than just the right side of the decision boundary.

## 5. Backpropagation and Optimization

- **Backpropagation:** For each mini-batch, `loss.backward()` uses reverse-mode
  automatic differentiation to compute the gradient of the loss with respect
  to every learnable parameter (weights and biases in every layer), applying
  the chain rule from the output layer back to the input layer.
- **Gradient descent / optimizer — Adam:** `optimizer.step()` updates each
  parameter using the Adam optimizer, which maintains per-parameter adaptive
  learning rates based on running estimates of the first and second moments
  of the gradients. This generally converges faster and is less sensitive to
  the initial learning rate choice than plain SGD, which is why it was
  selected here over vanilla stochastic gradient descent.
- **Learning rate:** 1e-3 initial, with `ReduceLROnPlateau` halving the rate
  whenever validation loss stalls for 5 epochs, allowing coarse progress
  early and finer convergence later.
- **Weight decay (L2 regularization):** 1e-4, added directly in the Adam
  optimizer to penalize large weights and reduce overfitting.

## 6. Training Loop and Overfitting Prevention

Each epoch:
1. Shuffle and batch the training data (batch size 32).
2. Forward pass → compute BCE loss → backward pass → Adam step (training only).
3. Evaluate the same-epoch model on the validation set (no gradient updates).
4. Log train/validation loss and accuracy.
5. Track the best validation loss seen so far; if 15 consecutive epochs pass
   without improvement, stop early.

The best-validation-loss checkpoint (not necessarily the final epoch's
weights) is restored before the final test evaluation, which protects
against reporting an overfit late-epoch model.

**Overfitting was monitored by comparing train vs. validation curves**
(see `docs/training_metrics.json`): training accuracy climbed into the
upper 90s while validation loss continued to fall and then plateaued
around epoch 45–85, at which point early stopping intervened — indicating
the regularization (dropout, weight decay, batch norm) combined with early
stopping successfully kept the train/validation gap small.

## 7. Final Results

| Split | Loss | Accuracy |
|---|---|---|
| Best validation checkpoint | 0.0466 | 97.9% |
| **Test (held-out, unseen)** | **0.0895** | **96.4%** |

The test accuracy of 96.4% on data the model never saw during training or
model selection indicates the network generalizes well and is not simply
memorizing the training set. The small gap between validation and test loss
is expected given the small overall dataset size (only 86 test samples) and
is consistent with a well-regularized model rather than overfitting.

## 8. Reproducibility

All randomness (data split, weight init, batch shuffling) is seeded
(`SEED = 42`), and the full per-epoch history plus final architecture
configuration is serialized to `docs/training_metrics.json` on every run of
`src/train.py`, so results can be re-verified or re-plotted without rerunning
training.
