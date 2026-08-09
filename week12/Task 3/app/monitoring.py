from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
MODEL_PATH = BASE_DIR / "model.joblib"
NEW_DATA_PATH = DATA_DIR / "new_data.csv"

ACCURACY_THRESHOLD = 0.90
DRIFT_THRESHOLD = 0.20


def load_reference_data():
    data = load_iris()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target


def calculate_drift(reference, current):
    scores = []
    for column in reference.columns:
        ref_mean = reference[column].mean()
        cur_mean = current[column].mean()
        scale = reference[column].std() or 1.0
        scores.append(abs(cur_mean - ref_mean) / scale)
    return float(np.mean(scores))


def main():
    reference, _ = load_reference_data()
    model = joblib.load(MODEL_PATH)

    if NEW_DATA_PATH.exists():
        current = pd.read_csv(NEW_DATA_PATH)
        target = current.pop("target").to_numpy()
    else:
        data = load_iris()
        current = pd.DataFrame(data.data, columns=data.feature_names)
        target = data.target

    predictions = model.predict(current)
    accuracy = accuracy_score(target, predictions)
    drift = calculate_drift(reference, current)

    retraining_required = accuracy < ACCURACY_THRESHOLD or drift > DRIFT_THRESHOLD

    print(f"accuracy={accuracy:.4f}")
    print(f"drift_score={drift:.4f}")
    print(f"retraining_required={retraining_required}")

    return retraining_required


if __name__ == "__main__":
    main()
