from app.retraining import should_retrain
from app.evaluate import classification_metrics, candidate_is_better


def test_accuracy_trigger():
    reasons = should_retrain(0.85, 0.08, 10, 0.05)
    assert "accuracy below threshold" in reasons


def test_new_data_trigger():
    reasons = should_retrain(0.95, 0.04, 100, 0.05)
    assert "enough new labeled data" in reasons


def test_metrics():
    result = classification_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert result["accuracy"] == 0.75
    assert result["recall"] == 0.5


def test_candidate_comparison():
    current = {"accuracy": 0.90, "f1": 0.80}
    candidate = {"accuracy": 0.92, "f1": 0.82}
    assert candidate_is_better(current, candidate)
