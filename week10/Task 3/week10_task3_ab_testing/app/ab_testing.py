from pathlib import Path
import csv
import hashlib
import math
import random
import time

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "ab_results.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "timestamp", "variant", "user_id", "prediction",
    "actual", "latency_ms", "correct"
]

def initialize_log():
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def assign_variant(user_id=None, traffic_to_b=0.5):
    if user_id:
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
    else:
        bucket = random.random()
    return "B" if bucket < traffic_to_b else "A"

def log_result(variant, user_id, prediction, actual, latency_ms):
    initialize_log()
    correct = ""
    if actual is not None:
        correct = int(int(actual) == int(prediction))

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variant": variant,
            "user_id": user_id or "",
            "prediction": int(prediction),
            "actual": "" if actual is None else int(actual),
            "latency_ms": round(float(latency_ms), 3),
            "correct": correct,
        })

def _rows():
    initialize_log()
    with LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def metrics():
    rows = _rows()
    result = {}
    for variant in ("A", "B"):
        subset = [r for r in rows if r["variant"] == variant]
        labeled = [r for r in subset if r["actual"] != ""]
        accuracy = None
        if labeled:
            accuracy = sum(int(r["correct"]) for r in labeled) / len(labeled)
        latencies = [float(r["latency_ms"]) for r in subset]
        result[variant] = {
            "requests": len(subset),
            "labeled_requests": len(labeled),
            "accuracy": accuracy,
            "average_latency_ms": (
                sum(latencies) / len(latencies) if latencies else None
            ),
        }
    return result

def two_proportion_z_test(success_a, total_a, success_b, total_b):
    if min(total_a, total_b) == 0:
        return None

    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)

    denominator = math.sqrt(
        pooled * (1 - pooled) * (1 / total_a + 1 / total_b)
    )
    if denominator == 0:
        return 0.0

    return (p1 - p2) / denominator
