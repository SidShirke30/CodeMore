from pathlib import Path
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

X, y = make_classification(
    n_samples=3000, n_features=20, n_informative=12,
    n_redundant=4, random_state=42
)
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

original = RandomForestClassifier(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)
optimized = RandomForestClassifier(
    n_estimators=80, max_depth=8, random_state=42, n_jobs=-1
)

original.fit(X_train, y_train)
optimized.fit(X_train, y_train)

joblib.dump(original, MODEL_DIR / "original_model.joblib")
joblib.dump(optimized, MODEL_DIR / "optimized_model.joblib", compress=3)
print("Models saved.")
