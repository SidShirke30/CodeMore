from pathlib import Path
import sys
import time
import joblib
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitoring import log_prediction, calculate_metrics

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError("model.pkl not found. Run: python app/train_model.py")

model = joblib.load(MODEL_PATH)
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    actual = payload.get("actual")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "'features' must contain exactly 4 values"}), 400

    try:
        features = [float(x) for x in features]
        if actual is not None:
            actual = int(actual)
    except (TypeError, ValueError):
        return jsonify({"error": "Features and actual label must be numeric"}), 400

    start = time.perf_counter()
    prediction = int(model.predict([features])[0])
    latency_ms = (time.perf_counter() - start) * 1000

    log_prediction(features, prediction, actual, latency_ms)
    return jsonify({
        "prediction": prediction,
        "latency_ms": round(latency_ms, 3)
    })

@app.get("/metrics")
def metrics():
    return jsonify(calculate_metrics())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
