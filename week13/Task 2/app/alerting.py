def evaluate_alerts(accuracy, latency_ms, psi, config):
    alerts = []

    if accuracy < config["accuracy_min"]:
        alerts.append(
            f"Accuracy degradation: {accuracy:.3f} < {config['accuracy_min']:.3f}"
        )

    if latency_ms > config["latency_max_ms"]:
        alerts.append(
            f"Latency degradation: {latency_ms:.1f} ms > {config['latency_max_ms']:.1f} ms"
        )

    if psi > config["psi_max"]:
        alerts.append(
            f"Data drift detected: PSI {psi:.3f} > {config['psi_max']:.3f}"
        )

    return alerts
