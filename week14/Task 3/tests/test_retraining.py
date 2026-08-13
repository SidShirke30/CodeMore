from app.retraining import check_triggers


def test_accuracy_trigger():
    config = {"accuracy_trigger": 0.80, "drift_trigger": 0.50}
    reasons = check_triggers({"accuracy": 0.70, "drift_score": 0.10}, config)
    assert "accuracy_below_threshold" in reasons


def test_drift_trigger():
    config = {"accuracy_trigger": 0.80, "drift_trigger": 0.50}
    reasons = check_triggers({"accuracy": 0.90, "drift_score": 0.80}, config)
    assert "data_drift_detected" in reasons
