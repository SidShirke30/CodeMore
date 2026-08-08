"""FastAPI model-serving application."""
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(
    title="Week 8 Task 3 - ML Prediction API",
    version="1.0.0",
    description="Production-style FastAPI endpoint serving a serialized scikit-learn model.",
)

model = None


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 numeric features from the Breast Cancer Wisconsin dataset.",
    )


@app.on_event("startup")
def load_model():
    """Load the serialized model once when the application starts."""
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Serialized model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Return class prediction and probabilities for one observation."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        X = np.asarray(request.features, dtype=float).reshape(1, -1)
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0].tolist()

        return {
            "prediction": prediction,
            "class_name": "malignant" if prediction == 0 else "benign",
            "probabilities": {
                "malignant": round(float(probabilities[0]), 6),
                "benign": round(float(probabilities[1]), 6),
            },
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid feature values: {exc}")
