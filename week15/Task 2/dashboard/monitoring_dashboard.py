from pathlib import Path
import csv
import matplotlib.pyplot as plt


DATA = Path(__file__).resolve().parents[1] / "data" / "metrics.csv"


def load_metrics():
    with DATA.open(newline="") as f:
        return list(csv.DictReader(f))


def create_dashboard():
    rows = load_metrics()
    steps = [r["timestamp"] for r in rows]
    accuracy = [float(r["accuracy"]) for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, accuracy, marker="o")
    plt.axhline(0.90, linestyle="--", label="Minimum accuracy")
    plt.title("Model Accuracy Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    output = Path(__file__).resolve().parents[1] / "reports" / "accuracy_dashboard.png"
    plt.savefig(output)
    print(f"Dashboard saved to {output}")


if __name__ == "__main__":
    create_dashboard()
