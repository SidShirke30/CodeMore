import time
from functools import wraps

from flask import request
from monitoring import log_prediction


def monitor_prediction_route(func):
    """Decorator that logs prediction output and API latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        prediction = None
        actual = request.headers.get("X-Actual-Label")

        try:
            payload = result.get_json(silent=True) or {}
            prediction = payload.get("prediction")
        except Exception:
            pass

        if prediction is not None:
            log_prediction(request.path, prediction, actual, latency_ms)

        return result

    return wrapper
