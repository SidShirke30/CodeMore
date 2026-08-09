from pathlib import Path
import csv
import math
import time
from collections import Counter

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
FIELDS = ["timestamp", "features", "prediction", "actual", "latency_ms", "correct"]

def initialize_log():
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def log_prediction(features, prediction, actual, latency_ms):
    initialize_log()
    correct = ""
    if actual is not None:
        correct = int(int(actual) == int(prediction))

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "features": "|".join(str(float(x)) for x in features),
            "prediction": int(prediction),
            "actual": "" if actual is None else int(actual),
            "latency_ms": round(float(latency_ms), 3),
            "correct": correct,
        })

def read_rows():
    initialize_log()
    with LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def calculate_metrics():
    rows = read_rows()
    if not rows:
        return {
            "request_count": 0,
            "accuracy": None,
            "average_latency_ms": None,
            "prediction_distribution": {},
        }

    predictions = [int(r["prediction"]) for r in rows]
    labeled = [r for r in rows if r["actual"] != ""]
    accuracy = None
    if labeled:
        accuracy = sum(int(r["correct"]) for r in labeled) / len(labeled)

    latency = [float(r["latency_ms"]) for r in rows]
    return {
        "request_count": len(rows),
        "accuracy": accuracy,
        "average_latency_ms": sum(latency) / len(latency),
        "prediction_distribution": dict(Counter(predictions)),
    }

def population_stability_index(reference_counts, current_counts):
    keys = set(reference_counts) | set(current_counts)
    ref_total = sum(reference_counts.values()) or 1
    cur_total = sum(current_counts.values()) or 1
    psi = 0.0

    for key in keys:
        ref = max(reference_counts.get(key, 0) / ref_total, 1e-6)
        cur = max(current_counts.get(key, 0) / cur_total, 1e-6)
        psi += (cur - ref) * math.log(cur / ref)

    return psi
