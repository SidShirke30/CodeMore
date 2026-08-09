from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "monitoring_history.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "monitoring_dashboard.png"

if not DATA_PATH.exists():
    print("No monitoring history found. Run monitoring jobs first.")
else:
    df = pd.read_csv(DATA_PATH)
    ax = df.plot(x="run", y=["accuracy", "drift_score"], marker="o")
    ax.set_title("Model Performance and Drift Over Time")
    ax.set_ylabel("Metric value")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f"Dashboard saved to {OUTPUT}")
