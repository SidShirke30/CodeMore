import csv
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
METRICS_FILE = os.path.join(LOG_DIR, "model_metrics.csv")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "api.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

FIELDS = [
    "timestamp", "endpoint", "prediction", "actual",
    "accuracy", "precision", "recall", "latency_ms"
]


def log_prediction(endpoint, prediction, actual=None, latency_ms=None):
    """Log a prediction and optional ground-truth label."""
    accuracy = precision = recall = ""

    if actual is not None:
        accuracy = float(accuracy_score([actual], [prediction]))
        precision = float(
            precision_score([actual], [prediction], average="weighted", zero_division=0)
        )
        recall = float(
            recall_score([actual], [prediction], average="weighted", zero_division=0)
        )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "prediction": prediction,
        "actual": "" if actual is None else actual,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "latency_ms": "" if latency_ms is None else round(float(latency_ms), 2),
    }

    exists = os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        "prediction endpoint=%s prediction=%s actual=%s latency_ms=%s",
        endpoint, prediction, actual, latency_ms
    )


def calculate_summary():
    """Calculate average production metrics from the monitoring log."""
    if not os.path.exists(METRICS_FILE):
        return {}

    df = pd.read_csv(METRICS_FILE)
    if df.empty:
        return {}

    summary = {}
    for metric in ["accuracy", "precision", "recall", "latency_ms"]:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not values.empty:
            summary[metric] = round(float(values.mean()), 4)
    return summary


def detect_degradation(accuracy_threshold=0.80, latency_threshold_ms=1000):
    """Return warnings when production performance crosses thresholds."""
    summary = calculate_summary()
    issues = []

    if summary.get("accuracy", 1.0) < accuracy_threshold:
        issues.append("Accuracy is below the configured threshold.")

    if summary.get("latency_ms", 0.0) > latency_threshold_ms:
        issues.append("Average prediction latency is above the configured threshold.")

    return issues


if __name__ == "__main__":
    examples = [
        ("predict", 1, 1, 120),
        ("predict", 0, 0, 135),
        ("predict", 1, 0, 180),
        ("predict", 1, 1, 145),
        ("predict", 0, 0, 130),
    ]

    for endpoint, prediction, actual, latency in examples:
        log_prediction(endpoint, prediction, actual, latency)

    print("Summary:", calculate_summary())
    print("Issues:", detect_degradation())
