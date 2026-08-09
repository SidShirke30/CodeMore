from pathlib import Path
import joblib
from sklearn.datasets import load_iris

MODEL_PATH = Path(__file__).resolve().parents[1] / "app" / "model.joblib"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)
    data = load_iris()
    prediction = model.predict(data.data[:3])
    assert len(prediction) == 3
