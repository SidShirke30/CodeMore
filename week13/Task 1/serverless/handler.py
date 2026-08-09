import json
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "app" / "model.joblib"
model = joblib.load(MODEL_PATH)


def handler(event, context=None):
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        features = body["features"]

        if not isinstance(features, list) or len(features) != 4:
            raise ValueError("features must contain exactly 4 values")

        values = np.asarray(features, dtype=float).reshape(1, -1)
        prediction = int(model.predict(values)[0])

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"prediction": prediction}),
        }

    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }
