from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or "features" not in payload:
        return jsonify({"error": "JSON body must contain 'features'"}), 400

    features = payload["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "features must be a list of 4 numeric values"}), 400

    try:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        prediction = int(model.predict(values)[0])
    except (TypeError, ValueError):
        return jsonify({"error": "features must contain numeric values"}), 400

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
