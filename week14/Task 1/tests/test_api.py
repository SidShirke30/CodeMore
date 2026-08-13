import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from app import app


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_prediction():
    response = app.test_client().post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert "prediction" in body
    assert "class_name" in body
    assert "probabilities" in body


def test_invalid_feature_count():
    response = app.test_client().post(
        "/predict",
        json={"features": [5.1, 3.5]}
    )
    assert response.status_code == 400
