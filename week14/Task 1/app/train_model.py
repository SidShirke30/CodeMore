from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
iris = load_iris()

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(iris.data, iris.target)

joblib.dump({
    "model": model,
    "feature_names": iris.feature_names,
    "class_names": iris.target_names.tolist(),
}, BASE_DIR / "model.joblib")

print("Model saved to app/model.joblib")
