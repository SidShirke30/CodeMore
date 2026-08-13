import os
import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

X, y = make_classification(
    n_samples=500, n_features=4, n_informative=3,
    n_redundant=0, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, os.path.join(MODEL_DIR, "current_model.joblib"))
print("Baseline model saved.")
