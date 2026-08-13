from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "ab_results.csv")
summary = df.groupby("model_version").agg(
    accuracy=("correct", "mean"),
    latency_ms=("latency_ms", "mean")
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
summary.plot.bar(x="model_version", y="accuracy", ax=axes[0], legend=False, title="Accuracy")
summary.plot.bar(x="model_version", y="latency_ms", ax=axes[1], legend=False, title="Latency (ms)")
plt.tight_layout()
out = ROOT / "reports" / "ab_dashboard.png"
plt.savefig(out, dpi=150)
print(f"Saved dashboard to {out}")
