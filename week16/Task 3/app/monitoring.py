from pathlib import Path
import pandas as pd

METRICS_FILE = Path(__file__).resolve().parents[1] / "data" / "metrics.csv"

def load_metrics(path=METRICS_FILE):
    return pd.read_csv(path)

def analyze_metrics(df, accuracy_threshold=0.85, latency_threshold_ms=250):
    issues = []
    if df["accuracy"].iloc[-1] < accuracy_threshold:
        issues.append("Accuracy has degraded below the configured threshold.")
    if df["latency_ms"].iloc[-1] > latency_threshold_ms:
        issues.append("Latency has exceeded the configured threshold.")
    if df["cpu_percent"].iloc[-1] > 80:
        issues.append("CPU utilization is high.")
    if df["memory_percent"].iloc[-1] > 80:
        issues.append("Memory utilization is high.")

    if len(df) >= 2 and df["accuracy"].iloc[-1] < df["accuracy"].iloc[0] - 0.05:
        issues.append("Potential model drift: accuracy dropped by more than 5 percentage points.")

    return issues

if __name__ == "__main__":
    metrics = load_metrics()
    print(metrics)
    print("\nDetected issues:")
    for issue in analyze_metrics(metrics):
        print(f"- {issue}")
