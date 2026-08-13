import pandas as pd
from app.ab_test import summarize, accuracy_z_test, choose_model

def test_summary_contains_models():
    df = pd.DataFrame({
        "model_version": ["A","A","B","B"],
        "correct": [1,0,1,1],
        "latency_ms": [10,12,8,9]
    })
    result = summarize(df)
    assert set(result.model_version) == {"A", "B"}

def test_z_test_returns_p_value():
    df = pd.DataFrame({
        "model_version": ["A"]*10 + ["B"]*10,
        "correct": [1,0,1,0,1,0,1,0,1,0] + [1,1,1,1,1,1,1,0,1,1],
        "latency_ms": [10]*20
    })
    result = accuracy_z_test(df)
    assert 0 <= result["p_value"] <= 1

def test_choose_model():
    summary = pd.DataFrame([
        {"model_version":"A","accuracy":0.8,"avg_latency_ms":120},
        {"model_version":"B","accuracy":0.9,"avg_latency_ms":100}
    ])
    test = {"p_value": 0.01, "accuracy_difference": 0.1}
    assert choose_model(summary, test) == "B"
