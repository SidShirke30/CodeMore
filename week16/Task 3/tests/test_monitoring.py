import pandas as pd
from app.monitoring import analyze_metrics

def test_accuracy_degradation_is_detected():
    df = pd.DataFrame({
        "accuracy": [0.94, 0.84],
        "latency_ms": [120, 275],
        "cpu_percent": [42, 83],
        "memory_percent": [48, 82],
    })
    issues = analyze_metrics(df)
    assert any("Accuracy" in issue for issue in issues)
    assert any("drift" in issue.lower() for issue in issues)
