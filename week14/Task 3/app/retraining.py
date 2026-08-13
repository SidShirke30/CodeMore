import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config", "retraining_config.json")
MODEL_PATH = os.path.join(ROOT, "models", "current_model.joblib")
CANDIDATE_PATH = os.path.join(ROOT, "models", "candidate_model.joblib")
NEW_DATA_PATH = os.path.join(ROOT, "data", "new_labeled_data.csv")
HISTORY_PATH = os.path.join(ROOT, "data", "retraining_history.csv")


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_data():
    df = pd.read_csv(NEW_DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
    }


def check_triggers(metrics, config):
    reasons = []
    if metrics.get("accuracy", 1.0) < config["accuracy_trigger"]:
        reasons.append("accuracy_below_threshold")

    if metrics.get("drift_score", 0.0) > config["drift_trigger"]:
        reasons.append("data_drift_detected")

    return reasons


def retrain_if_needed():
    config = load_config()
    X_train, X_test, y_train, y_test = load_data()

    if os.path.exists(MODEL_PATH):
        current = joblib.load(MODEL_PATH)
    else:
        current = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    current_metrics = evaluate(current, X_test, y_test)

    # A simple reproducible drift proxy: normalized change in feature means.
    drift_score = float(np.mean(np.abs(X_train.mean().values)))
    current_metrics["drift_score"] = round(drift_score, 4)
    reasons = check_triggers(current_metrics, config)

    candidate_metrics = {}
    promoted = False

    if reasons:
        candidate = LogisticRegression(max_iter=1000)
        candidate.fit(X_train, y_train)
        candidate_metrics = evaluate(candidate, X_test, y_test)
        candidate_metrics["drift_score"] = current_metrics["drift_score"]
        joblib.dump(candidate, CANDIDATE_PATH)

        improvement = candidate_metrics["accuracy"] - current_metrics["accuracy"]
        if improvement >= config["minimum_improvement"]:
            joblib.dump(candidate, MODEL_PATH)
            promoted = True

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggered": bool(reasons),
        "reasons": ";".join(reasons),
        "current_accuracy": current_metrics["accuracy"],
        "candidate_accuracy": candidate_metrics.get("accuracy", ""),
        "drift_score": current_metrics["drift_score"],
        "promoted": promoted,
    }

    history = pd.DataFrame([row])
    if os.path.exists(HISTORY_PATH):
        history.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        history.to_csv(HISTORY_PATH, index=False)

    return row


if __name__ == "__main__":
    print(retrain_if_needed())
