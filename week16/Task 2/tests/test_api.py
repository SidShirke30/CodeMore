import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_predict(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "class_name" in response.json


def test_invalid_features(client):
    response = client.post(
        "/predict",
        json={"features": [1, 2]},
    )
    assert response.status_code == 400
