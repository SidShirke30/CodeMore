from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "production_metrics.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

plt.figure(figsize=(9, 5))
plt.plot(df["timestamp"], df["accuracy"], marker="o")
plt.axhline(0.85, linestyle="--", label="Accuracy threshold")
plt.title("Model Accuracy Over Time")
plt.xlabel("Date")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "reports" / "accuracy_over_time.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(df["timestamp"], df["latency_ms"], marker="o")
plt.axhline(250, linestyle="--", label="Latency threshold")
plt.title("Inference Latency Over Time")
plt.xlabel("Date")
plt.ylabel("Latency (ms)")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "reports" / "latency_over_time.png", dpi=150)
plt.close()

print("Dashboard charts saved in reports/.")
