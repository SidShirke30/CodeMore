from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "model.joblib"
DATA_PATH = ROOT / "data" / "sample_data.csv"

def main():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
    model.fit(train_X, train_y)

    predictions = model.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)
    print(f"Test accuracy: {accuracy:.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_names": iris.feature_names,
            "class_names": iris.target_names.tolist(),
        },
        MODEL_PATH,
    )

    sample = X.copy()
    sample["target"] = y
    sample.to_csv(DATA_PATH, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Dataset saved to: {DATA_PATH}")

if __name__ == "__main__":
    main()
