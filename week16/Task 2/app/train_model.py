from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

data = load_iris()
model = LogisticRegression(max_iter=500)
model.fit(data.data, data.target)

joblib.dump(
    {
        "model": model,
        "target_names": data.target_names.tolist(),
    },
    MODEL_PATH,
)

print(f"Saved model to {MODEL_PATH}")
