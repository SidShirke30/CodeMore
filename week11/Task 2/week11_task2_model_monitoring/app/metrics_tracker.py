from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "performance.csv"


def calculate_metrics(y_true, y_pred):
    """Calculate standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def append_metrics(metrics, period="unknown"):
    """Append one metric record to the monitoring CSV."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "period": period,
        **metrics,
    }])

    if LOG_FILE.exists():
        row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_FILE, index=False)


def load_metrics():
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=["timestamp", "period", "accuracy", "precision", "recall"]
        )
    return pd.read_csv(LOG_FILE)


if __name__ == "__main__":
    sample = calculate_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    append_metrics(sample, period="sample")
    print(sample)
