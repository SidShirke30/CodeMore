from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
DATA_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

X, y = make_classification(
    n_samples=3000, n_features=20, n_informative=12,
    n_redundant=4, random_state=42
)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rows = []
for filename in ["original_model.joblib", "optimized_model.joblib"]:
    model = joblib.load(MODEL_DIR / filename)
    model.predict(X_test[:10])
    start = time.perf_counter()
    pred = model.predict(X_test)
    elapsed = time.perf_counter() - start
    rows.append({
        "model": filename.replace("_model.joblib", ""),
        "accuracy": accuracy_score(y_test, pred),
        "latency_ms_total": elapsed * 1000,
        "latency_ms_per_prediction": elapsed * 1000 / len(X_test),
        "model_size_kb": (MODEL_DIR / filename).stat().st_size / 1024,
    })

df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "benchmark_results.csv", index=False)

o = df.iloc[0]
p = df.iloc[1]
report = (
    "# Model Optimization Report\n\n"
    "## Results\n\n"
    "| Metric | Original | Optimized |\n"
    "|---|---:|---:|\n"
    f"| Accuracy | {o['accuracy']:.4f} | {p['accuracy']:.4f} |\n"
    f"| Total latency (ms) | {o['latency_ms_total']:.3f} | {p['latency_ms_total']:.3f} |\n"
    f"| Latency / prediction (ms) | {o['latency_ms_per_prediction']:.5f} | {p['latency_ms_per_prediction']:.5f} |\n"
    f"| Model size (KB) | {o['model_size_kb']:.2f} | {p['model_size_kb']:.2f} |\n\n"
    "## Interpretation\n\n"
    f"- Accuracy change: **{p['accuracy'] - o['accuracy']:+.4f}**\n"
    f"- Latency change: **{(p['latency_ms_per_prediction']/o['latency_ms_per_prediction']-1)*100:+.2f}%**\n"
    f"- Model-size change: **{(p['model_size_kb']/o['model_size_kb']-1)*100:+.2f}%**\n\n"
    "The optimized model uses fewer trees and a maximum tree depth of 8, reducing model complexity. "
    "Joblib compression further reduces the serialized artifact size.\n\n"
    "## Deployment Recommendation\n\n"
    "Accept the optimized model only if its accuracy remains within the required business threshold. "
    "Validate it on production-like data before replacing the deployed model.\n"
)
(REPORT_DIR / "optimization_report.md").write_text(report, encoding="utf-8")
print(df.to_string(index=False))
