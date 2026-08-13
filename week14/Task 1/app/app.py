from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
bundle = joblib.load(BASE_DIR / "model.joblib")

model = bundle["model"]
feature_names = bundle["feature_names"]
class_names = bundle["class_names"]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    features = payload.get("features")

    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list"}), 400

    if len(features) != len(feature_names):
        return jsonify({
            "error": f"Expected {len(feature_names)} features",
            "feature_names": feature_names
        }), 400

    try:
        values = np.asarray(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numeric"}), 400

    prediction = int(model.predict(values)[0])
    probabilities = model.predict_proba(values)[0].tolist()

    return jsonify({
        "prediction": prediction,
        "class_name": class_names[prediction],
        "probabilities": probabilities
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
