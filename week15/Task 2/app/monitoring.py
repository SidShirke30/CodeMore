from metrics_collector import ModelMetrics
from anomaly_detector import detect_anomalies
from alerting import trigger_alert


THRESHOLDS = {
    "min_accuracy": 0.90,
    "max_latency_ms": 250,
    "max_cpu_percent": 85,
    "max_memory_percent": 85,
    "accuracy_drop": 0.05,
}


def run_monitoring():
    current = ModelMetrics(
        accuracy=0.87,
        latency_ms=310,
        cpu_percent=72,
        memory_percent=61,
        error_rate=0.13,
    )

    baseline = {"accuracy": 0.94}
    alerts = detect_anomalies(current, baseline, THRESHOLDS)
    return trigger_alert(alerts)


if __name__ == "__main__":
    run_monitoring()
