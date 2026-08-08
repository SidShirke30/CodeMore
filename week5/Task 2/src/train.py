"""
train.py

Trains the CNN defined in cnn_model.py on MNIST (see dataset.py for the
data pipeline), tracks training/validation loss & accuracy across epochs,
and evaluates the final model on the held-out test set — producing a
confusion matrix, classification report, and loss/accuracy curve plots.

Usage:
    python src/train.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from cnn_model import build_model
from dataset import CLASS_NAMES, get_dataloaders

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 6  # early stopping patience (epochs without val improvement)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Subsetting keeps training tractable on CPU-only environments while still
# giving a meaningful, representative result. Set to None to use the full
# 60,000/10,000 train/test images given a GPU or more time.
TRAIN_SUBSET = 20000
TEST_SUBSET = 5000

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_DIR = os.path.join(ROOT, "docs")
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "saved_cnn_model.pth")
METRICS_PATH = os.path.join(DOCS_DIR, "training_metrics.json")
LOSS_CURVE_PATH = os.path.join(DOCS_DIR, "loss_accuracy_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(DOCS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(DOCS_DIR, "classification_report.txt")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += yb.size(0)
            total_loss += loss.item() * yb.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(yb.numpy().tolist())
    return np.array(all_targets), np.array(all_preds)


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — MNIST Test Set")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    set_seed(SEED)
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        train_subset=TRAIN_SUBSET,
        test_subset=TEST_SUBSET,
        seed=SEED,
    )

    model = build_model(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    print(f"Training on device: {DEVICE}")
    print(f"Train/Val/Test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
    print("-" * 70)
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

    # Confusion matrix + classification report on test set
    y_true, y_pred = collect_predictions(model, test_loader)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    print("\nClassification Report:\n", report)

    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PATH)
    plot_curves(history, LOSS_CURVE_PATH)
    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(report)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel weights saved to {MODEL_SAVE_PATH}")

    results = {
        "architecture": "CNN: 3 conv blocks (32/64/128 channels) + 2 dense layers",
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "train_subset": TRAIN_SUBSET,
        "test_subset": TEST_SUBSET,
        "history": history,
        "confusion_matrix": cm.tolist(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
