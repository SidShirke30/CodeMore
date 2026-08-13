import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
HISTORY = os.path.join(ROOT, "data", "retraining_history.csv")
OUTPUT = os.path.join(ROOT, "dashboard", "retraining_history.png")


def create_dashboard():
    if not os.path.exists(HISTORY):
        raise FileNotFoundError("Run app/retraining.py first.")

    df = pd.read_csv(HISTORY)
    df["current_accuracy"] = pd.to_numeric(df["current_accuracy"], errors="coerce")
    df["candidate_accuracy"] = pd.to_numeric(df["candidate_accuracy"], errors="coerce")

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["current_accuracy"], marker="o", label="Current")
    plt.plot(df.index, df["candidate_accuracy"], marker="o", label="Candidate")
    plt.xlabel("Retraining event")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Across Retraining Events")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"Dashboard saved to {OUTPUT}")


if __name__ == "__main__":
    create_dashboard()
