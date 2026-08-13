from dataclasses import dataclass


@dataclass
class RetrainingConfig:
    min_accuracy: float = 0.90
    max_error_rate: float = 0.10
    min_new_samples: int = 100
    drift_score: float = 0.20


def should_retrain(accuracy, error_rate, new_samples, drift_score, config=None):
    config = config or RetrainingConfig()

    reasons = []

    if accuracy < config.min_accuracy:
        reasons.append("accuracy below threshold")

    if error_rate > config.max_error_rate:
        reasons.append("error rate above threshold")

    if new_samples >= config.min_new_samples:
        reasons.append("enough new labeled data")

    if drift_score > config.drift_score:
        reasons.append("data drift detected")

    return reasons
