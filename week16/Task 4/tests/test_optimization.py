from sklearn.ensemble import RandomForestClassifier

def test_optimized_configuration_reduces_complexity():
    original = RandomForestClassifier(n_estimators=200, max_depth=None)
    optimized = RandomForestClassifier(n_estimators=80, max_depth=8)
    assert optimized.n_estimators < original.n_estimators
    assert optimized.max_depth == 8
