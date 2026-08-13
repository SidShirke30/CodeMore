def detect_anomalies(metrics, baseline, thresholds):
    alerts = []

    if metrics.accuracy < thresholds["min_accuracy"]:
        alerts.append("Accuracy below threshold")

    if metrics.latency_ms > thresholds["max_latency_ms"]:
        alerts.append("Latency above threshold")

    if metrics.cpu_percent > thresholds["max_cpu_percent"]:
        alerts.append("CPU utilization above threshold")

    if metrics.memory_percent > thresholds["max_memory_percent"]:
        alerts.append("Memory utilization above threshold")

    if baseline and metrics.accuracy < baseline["accuracy"] - thresholds["accuracy_drop"]:
        alerts.append("Accuracy degradation detected")

    return alerts
