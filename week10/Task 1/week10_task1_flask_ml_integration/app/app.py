from pathlib import Path
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 415

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    features = payload.get("features")

    if not isinstance(features, list):
        return jsonify({
            "error": "'features' must be a JSON list"
        }), 400

    if len(features) != 4:
        return jsonify({
            "error": "'features' must contain exactly 4 numeric values"
        }), 400

    try:
        numeric_features = [float(value) for value in features]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    try:
        prediction = int(model.predict([numeric_features])[0])
        response = {
            "prediction": prediction
        }

        if hasattr(model, "predict_proba"):
            response["probabilities"] = [
                round(float(value), 6)
                for value in model.predict_proba([numeric_features])[0]
            ]

        return jsonify(response)

    except Exception as exc:
        return jsonify({
            "error": "Prediction failed",
            "details": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
