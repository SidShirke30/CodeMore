from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "performance.csv"
OUTPUT = Path(__file__).resolve().parent / "performance_metrics.png"

df = pd.read_csv(LOG_FILE)

plt.figure(figsize=(9, 5))
plt.plot(df["period"], df["accuracy"], marker="o", label="Accuracy")
plt.plot(df["period"], df["precision"], marker="o", label="Precision")
plt.plot(df["period"], df["recall"], marker="o", label="Recall")
plt.xlabel("Monitoring Period")
plt.ylabel("Metric")
plt.title("Image Classification Model Performance Over Time")
plt.ylim(0, 1)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"Dashboard saved to: {OUTPUT}")
