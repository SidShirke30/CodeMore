import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_missing_image():
    client = app.test_client()
    response = client.post("/predict")
    assert response.status_code == 400


def test_prediction():
    client = app.test_client()
    image = Image.new("L", (8, 8), color=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        data={"image": (buffer, "digit.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert "predicted_class" in response.json
