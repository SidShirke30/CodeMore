import json
from pathlib import Path
import numpy as np
import pandas as pd

from alerting import evaluate_alerts

ROOT = Path(__file__).resolve().parents[1]


def calculate_psi(expected, actual, bins=5):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = (expected_counts + 1e-6) / (expected_counts.sum() + 1e-6 * len(expected_counts))
    actual_pct = (actual_counts + 1e-6) / (actual_counts.sum() + 1e-6 * len(actual_counts))

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring():
    with open(ROOT / "config" / "monitoring_config.json", encoding="utf-8") as f:
        config = json.load(f)

    baseline = pd.read_csv(ROOT / "data" / "baseline.csv")
    production = pd.read_csv(ROOT / "data" / "production_metrics.csv")

    latest = production.iloc[-1]
    psi_values = [
        calculate_psi(baseline[col], production[col])
        for col in ["feature_1", "feature_2", "feature_3", "feature_4"]
    ]
    max_psi = max(psi_values)

    alerts = evaluate_alerts(
        float(latest["accuracy"]),
        float(latest["latency_ms"]),
        max_psi,
        config,
    )

    print(f"Latest accuracy: {latest['accuracy']:.3f}")
    print(f"Latest latency: {latest['latency_ms']:.1f} ms")
    print(f"Maximum PSI drift: {max_psi:.3f}")

    if alerts:
        print("\nALERTS")
        for alert in alerts:
            print(f"- {alert}")
    else:
        print("\nNo alerts detected.")


if __name__ == "__main__":
    run_monitoring()
