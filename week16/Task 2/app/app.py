import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
target_names = bundle["target_names"]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not payload or "features" not in payload:
        return jsonify({"error": "JSON body must contain 'features'"}), 400

    features = payload["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "features must be a list of 4 numeric values"}), 400

    try:
        values = [float(value) for value in features]
    except (TypeError, ValueError):
        return jsonify({"error": "all features must be numeric"}), 400

    prediction = int(model.predict([values])[0])

    return jsonify({
        "prediction": prediction,
        "class_name": target_names[prediction],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
