"""
train.py

Trains and compares three recurrent architectures (vanilla RNN, LSTM, GRU)
on the Daily Minimum Temperatures time series forecasting task. For each
model, tracks training/validation loss and gradient norms per epoch (to
surface vanishing/exploding gradient behavior), then evaluates on the
held-out test set using RMSE and MAE, and generates comparison plots.

Usage:
    python src/train.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from preprocess import WINDOW_SIZE, get_dataloaders
from rnn_model import build_model

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
PATIENCE = 10
GRAD_CLIP = 1.0
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPES = ["rnn", "lstm", "gru"]

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS_DIR = os.path.join(ROOT, "docs")
MODELS_DIR = os.path.join(ROOT, "models")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_grad_norm(model) -> float:
    """L2 norm of gradients across all parameters, computed after backward()."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def run_epoch(model, loader, criterion, optimizer=None, clip: float = None, track_grad_norm: bool = False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    grad_norms = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                loss.backward()
                if track_grad_norm:
                    grad_norms.append(compute_grad_norm(model))
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            n_samples += yb.size(0)

    avg_loss = total_loss / n_samples
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else None
    return avg_loss, avg_grad_norm


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        preds = model(xb).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.numpy())
    return np.concatenate(all_targets).flatten(), np.concatenate(all_preds).flatten()


def train_one_model(cell_type: str, train_loader, val_loader):
    set_seed(SEED)
    model = build_model(cell_type=cell_type, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history = {"train_loss": [], "val_loss": [], "grad_norm": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    print(f"\n{'=' * 70}\nTraining {cell_type.upper()}\n{'=' * 70}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, grad_norm = run_epoch(
            model, train_loader, criterion, optimizer, clip=GRAD_CLIP, track_grad_norm=True
        )
        val_loss, _ = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(grad_norm)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.5f} | "
                  f"val_loss={val_loss:.5f} | avg_grad_norm={grad_norm:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_loss


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def plot_loss_curves(all_histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, hist in all_histories.items():
        axes[0].plot(hist["train_loss"], label=f"{name.upper()} train")
        axes[0].plot(hist["val_loss"], label=f"{name.upper()} val", linestyle="--")
    axes[0].set_title("Training / Validation Loss (MSE, scaled)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, hist in all_histories.items():
        axes[1].plot(hist["grad_norm"], label=name.upper())
    axes[1].set_title("Average Gradient Norm per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L2 Gradient Norm")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(dates_test, y_true, preds_by_model, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_test, y_true, label="Actual", color="black", linewidth=1.5)
    for name, preds in preds_by_model.items():
        ax.plot(dates_test, preds, label=f"{name.upper()} predicted", alpha=0.8, linewidth=1.2)
    ax.set_title("Test Set: Actual vs. Predicted Daily Minimum Temperature")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    set_seed(SEED)
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, scaler, series = get_dataloaders(
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    all_histories = {}
    all_metrics = {}
    preds_by_model_original_scale = {}
    y_true_original_scale = None

    for cell_type in MODEL_TYPES:
        model, history, best_val_loss = train_one_model(cell_type, train_loader, val_loader)
        all_histories[cell_type] = history

        y_true_scaled, y_pred_scaled = collect_predictions(model, test_loader)

        # Inverse-transform back to original temperature scale (°C) for interpretable metrics
        y_true_orig = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        test_rmse = rmse(y_true_orig, y_pred_orig)
        test_mae = mae(y_true_orig, y_pred_orig)

        print(f"[{cell_type.upper()}] Test RMSE: {test_rmse:.3f} °C | Test MAE: {test_mae:.3f} °C")

        all_metrics[cell_type] = {
            "best_val_loss_scaled_mse": best_val_loss,
            "test_rmse_celsius": test_rmse,
            "test_mae_celsius": test_mae,
            "epochs_run": len(history["train_loss"]),
            "final_grad_norm": history["grad_norm"][-1],
            "max_grad_norm": max(g for g in history["grad_norm"] if g is not None),
            "min_grad_norm": min(g for g in history["grad_norm"] if g is not None),
        }

        preds_by_model_original_scale[cell_type] = y_pred_orig
        if y_true_original_scale is None:
            y_true_original_scale = y_true_orig

        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"saved_{cell_type}_model.pth"))

    # Dates aligned with the test predictions (offset by window size, handled in preprocess)
    n_test_preds = len(y_true_original_scale)
    dates_test = series.index[-n_test_preds:]

    plot_loss_curves(all_histories, os.path.join(DOCS_DIR, "training_history.png"))
    plot_predictions(dates_test, y_true_original_scale, preds_by_model_original_scale,
                      os.path.join(DOCS_DIR, "prediction_comparison.png"))

    results = {
        "dataset": "Daily Minimum Temperatures, Melbourne 1981-1990",
        "window_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "metrics": all_metrics,
        "histories": all_histories,
    }
    with open(os.path.join(DOCS_DIR, "training_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Summary (test set, original °C scale):")
    for cell_type, m in all_metrics.items():
        print(f"  {cell_type.upper():5s} | RMSE: {m['test_rmse_celsius']:.3f} | MAE: {m['test_mae_celsius']:.3f}")
    print(f"\nAll artifacts saved to {DOCS_DIR} and {MODELS_DIR}")


if __name__ == "__main__":
    main()
