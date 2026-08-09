from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def preprocess_image(file_storage):
    try:
        image = Image.open(file_storage).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file.") from exc

    image = image.resize((8, 8))
    pixels = np.asarray(image, dtype=np.float32)

    # sklearn digits uses approximately 0..16 intensity values.
    pixels = pixels / 255.0 * 16.0
    return pixels.reshape(1, -1)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file in request."}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        features = preprocess_image(image_file)
        prediction = int(model.predict(features)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

        response = {"predicted_class": prediction}
        if confidence is not None:
            response["confidence"] = round(confidence, 4)

        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
