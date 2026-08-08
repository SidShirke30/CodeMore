"""Generate the serialized model used by the FastAPI application."""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

data = load_breast_cancer()
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
)
model.fit(data.data, data.target)
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
