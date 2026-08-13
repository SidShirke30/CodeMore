from app.monitoring import detect_degradation


def test_degradation_returns_list():
    assert isinstance(detect_degradation(), list)


def test_monitoring_file_exists():
    from pathlib import Path
    assert Path("logs/model_metrics.csv").exists()
