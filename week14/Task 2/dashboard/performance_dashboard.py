import os
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
METRICS_FILE = os.path.join(ROOT, "logs", "model_metrics.csv")
OUTPUT = os.path.join(ROOT, "dashboard", "performance_dashboard.png")


def create_dashboard():
    if not os.path.exists(METRICS_FILE):
        raise FileNotFoundError("Run app/monitoring.py first.")

    df = pd.read_csv(METRICS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for col in ["accuracy", "precision", "recall", "latency_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(df["timestamp"], df["accuracy"], marker="o", label="Accuracy")
    axes[0].plot(df["timestamp"], df["precision"], marker="o", label="Precision")
    axes[0].plot(df["timestamp"], df["recall"], marker="o", label="Recall")
    axes[0].set_title("Model Performance")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["timestamp"], df["latency_ms"], marker="o")
    axes[1].set_title("Prediction Latency")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(f"Dashboard saved to {OUTPUT}")


if __name__ == "__main__":
    create_dashboard()
