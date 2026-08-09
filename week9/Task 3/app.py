import argparse
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


def train_and_save_model():
    iris = load_iris()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(iris.data, iris.target)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


model = load_model()
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    features = payload.get("features")

    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list"}), 400

    if len(features) != 4:
        return jsonify({"error": "'features' must contain exactly 4 values"}), 400

    try:
        values = [float(x) for x in features]
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric"}), 400

    sample = np.array(values, dtype=float).reshape(1, -1)
    prediction = int(model.predict(sample)[0])

    response = {"prediction": prediction}

    if hasattr(model, "predict_proba"):
        response["probability"] = float(np.max(model.predict_proba(sample)[0]))

    return jsonify(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain and overwrite model.pkl before starting Flask.",
    )
    args = parser.parse_args()

    if args.train:
        train_and_save_model()

    app.run(host="127.0.0.1", port=5000, debug=False)
