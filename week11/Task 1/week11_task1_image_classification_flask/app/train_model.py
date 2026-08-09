from pathlib import Path

import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42,
    stratify=digits.target,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
])

model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
