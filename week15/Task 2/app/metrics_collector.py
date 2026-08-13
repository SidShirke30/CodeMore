from dataclasses import dataclass


@dataclass
class ModelMetrics:
    accuracy: float
    latency_ms: float
    cpu_percent: float
    memory_percent: float
    error_rate: float


def calculate_metrics(y_true, y_pred, latencies, cpu_percent, memory_percent):
    if not y_true:
        raise ValueError("y_true cannot be empty")

    correct = sum(a == b for a, b in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    error_rate = 1 - accuracy
    avg_latency = sum(latencies) / len(latencies)

    return ModelMetrics(
        accuracy=accuracy,
        latency_ms=avg_latency,
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        error_rate=error_rate,
    )
