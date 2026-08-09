from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = "app/model.joblib"
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not payload or "features" not in payload:
        return jsonify({"error": "JSON body must contain 'features'"}), 400

    features = payload["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "features must be a list of exactly 4 numeric values"}), 400

    try:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        prediction = model.predict(values)[0]
        return jsonify({"prediction": int(prediction)})
    except (ValueError, TypeError):
        return jsonify({"error": "features must contain numeric values"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
