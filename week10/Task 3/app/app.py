from pathlib import Path
import sys
import time
import joblib
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_testing import assign_variant, log_result, metrics

ROOT = Path(__file__).resolve().parent
model_a = joblib.load(ROOT / "model_a.pkl")
model_b = joblib.load(ROOT / "model_b.pkl")

app = Flask(__name__)

TRAFFIC_TO_B = 0.5

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "traffic_to_b": TRAFFIC_TO_B
    })

@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    user_id = payload.get("user_id")
    actual = payload.get("actual")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "'features' must contain exactly 4 values"}), 400

    try:
        features = [float(x) for x in features]
        if actual is not None:
            actual = int(actual)
    except (TypeError, ValueError):
        return jsonify({"error": "Features and actual must be numeric"}), 400

    variant = assign_variant(user_id, TRAFFIC_TO_B)
    model = model_b if variant == "B" else model_a

    start = time.perf_counter()
    prediction = int(model.predict([features])[0])
    latency_ms = (time.perf_counter() - start) * 1000

    log_result(variant, user_id, prediction, actual, latency_ms)

    return jsonify({
        "variant": variant,
        "prediction": prediction,
        "latency_ms": round(latency_ms, 3)
    })

@app.get("/metrics")
def get_metrics():
    return jsonify(metrics())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
