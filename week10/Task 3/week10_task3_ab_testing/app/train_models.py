from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent

iris = load_iris()

model_a = RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    random_state=42
)
model_b = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42
)

model_a.fit(iris.data, iris.target)
model_b.fit(iris.data, iris.target)

joblib.dump(model_a, ROOT / "model_a.pkl")
joblib.dump(model_b, ROOT / "model_b.pkl")

print("Saved model_a.pkl and model_b.pkl")
