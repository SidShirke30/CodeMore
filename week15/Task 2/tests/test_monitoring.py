from app.metrics_collector import calculate_metrics
from app.anomaly_detector import detect_anomalies


def test_metric_calculation():
    metrics = calculate_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [100, 120, 110, 130],
        50,
        40,
    )
    assert metrics.accuracy == 0.75
    assert metrics.latency_ms == 115


def test_accuracy_alert():
    metrics = calculate_metrics([1, 1], [1, 0], [100, 100], 50, 40)
    alerts = detect_anomalies(
        metrics,
        {"accuracy": 0.95},
        {
            "min_accuracy": 0.90,
            "max_latency_ms": 250,
            "max_cpu_percent": 85,
            "max_memory_percent": 85,
            "accuracy_drop": 0.05,
        },
    )
    assert alerts
