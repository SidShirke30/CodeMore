from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from metrics_tracker import calculate_metrics, append_metrics, load_metrics

ACCURACY_DROP_THRESHOLD = 0.10


def simulate_monitoring():
    """Simulate several monitoring periods with gradual model decay."""
    rng = np.random.default_rng(42)

    periods = [
        ("week_1", 0.92),
        ("week_2", 0.90),
        ("week_3", 0.87),
        ("week_4", 0.78),
        ("week_5", 0.74),
    ]

    for period, expected_accuracy in periods:
        y_true = rng.integers(0, 2, size=200)
        y_pred = y_true.copy()

        errors = int((1 - expected_accuracy) * len(y_true))
        error_indices = rng.choice(len(y_true), size=errors, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]

        metrics = calculate_metrics(y_true, y_pred)
        append_metrics(metrics, period=period)
        print(f"{period}: {metrics}")


def detect_model_decay():
    df = load_metrics()

    if len(df) < 2:
        print("Not enough monitoring periods for decay detection.")
        return

    baseline = df.iloc[0]["accuracy"]
    latest = df.iloc[-1]["accuracy"]
    drop = baseline - latest

    print(f"Baseline accuracy: {baseline:.3f}")
    print(f"Latest accuracy:   {latest:.3f}")
    print(f"Accuracy change:   {-drop:.3f}")

    if drop >= ACCURACY_DROP_THRESHOLD:
        print("WARNING: Potential model decay detected.")
    else:
        print("Model performance is within the configured threshold.")


if __name__ == "__main__":
    simulate_monitoring()
    detect_model_decay()
