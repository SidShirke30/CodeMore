"""
train.py

Loads the Breast Cancer Wisconsin (Diagnostic) tabular dataset, trains an
MLP binary classifier defined in model.py, monitors train/validation
loss & accuracy across epochs, and evaluates the final model on a held-out
test set.

Usage:
    python src/train.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import build_model

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15  # early stopping patience (epochs without val improvement)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "training_metrics.json")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model.pt")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data():
    """Load and split the breast cancer dataset (569 samples, 30 features, binary target)."""
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(np.float32)

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def to_tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    return (
        to_tensors(X_train, y_train),
        to_tensors(X_val, y_val),
        to_tensors(X_test, y_test),
        data.feature_names,
    )


def batches(X, y, batch_size, shuffle=True):
    n = X.size(0)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        yield X[b], y[b]


def compute_accuracy(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item()


def run_epoch(model, X, y, criterion, optimizer=None):
    """Runs one epoch. If optimizer is given, trains; otherwise evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for xb, yb in batches(X, y, BATCH_SIZE, shuffle=is_train):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                loss.backward()          # backpropagation
                optimizer.step()         # gradient descent step (Adam)

            total_loss += loss.item()
            total_acc += compute_accuracy(logits, yb)
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def main():
    set_seed(SEED)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = load_data()

    input_dim = X_train.shape[1]
    model = build_model(input_dim=input_dim, output_dim=1).to(DEVICE)

    # Binary classification -> BCEWithLogitsLoss (combines Sigmoid + BCE, numerically stable)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    print(f"Training on device: {DEVICE}")
    print(f"Train/Val/Test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, X_train, y_train, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, X_val, y_val, criterion, optimizer=None)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping to prevent overfitting
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch} "
                      f"(no val improvement for {PATIENCE} epochs).")
                break

    # Restore best weights before final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, X_test, y_test, criterion, optimizer=None)
    print("-" * 70)
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")

    # Save metrics for the architecture report
    results = {
        "input_dim": input_dim,
        "hidden_dims": [64, 32, 16],
        "output_dim": 1,
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "history": history,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training metrics saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
