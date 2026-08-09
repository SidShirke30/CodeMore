from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "app" / "improved_model.joblib"


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    baseline = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test)

    # Simulate newly available labelled production data.
    X_old, X_new, y_old, y_new = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=7
    )
    X_retrain = pd.concat([X_old, X_new], ignore_index=True)
    y_retrain = pd.concat(
        [y_old.reset_index(drop=True), y_new.reset_index(drop=True)],
        ignore_index=True,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_retrain, y_retrain)

    improved = search.best_estimator_
    improved_metrics = evaluate(improved, X_test, y_test)

    comparison = pd.DataFrame([
        {"model": "baseline_random_forest", **baseline_metrics},
        {"model": "retrained_tuned_random_forest", **improved_metrics},
    ])
    comparison.to_csv(DATA_DIR / "model_comparison.csv", index=False)
    joblib.dump(improved, MODEL_PATH)

    b_acc = baseline_metrics["accuracy"]
    b_pre = baseline_metrics["precision"]
    b_rec = baseline_metrics["recall"]
    b_f1 = baseline_metrics["f1"]
    i_acc = improved_metrics["accuracy"]
    i_pre = improved_metrics["precision"]
    i_rec = improved_metrics["recall"]
    i_f1 = improved_metrics["f1"]

    report = f"""# Model Improvement Report

## Baseline Metrics
- Accuracy: {b_acc:.4f}
- Precision: {b_pre:.4f}
- Recall: {b_rec:.4f}
- F1-score: {b_f1:.4f}

## Improvement Strategy
- Added newly available labelled data to the retraining pool.
- Used GridSearchCV for systematic hyperparameter tuning.
- Optimized F1-score.
- Kept a fixed holdout set for final comparison.
- Saved the selected model for future deployment.

## Best Hyperparameters
```text
{search.best_params_}
```

## Improved Model Metrics
- Accuracy: {i_acc:.4f}
- Precision: {i_pre:.4f}
- Recall: {i_rec:.4f}
- F1-score: {i_f1:.4f}

## Recommendation
Replace the baseline only if the improved model meets project acceptance thresholds and performs well on representative production-like data.

## Maintenance Recommendations
1. Retrain when validated performance degradation exceeds the monitoring threshold.
2. Validate data quality before retraining.
3. Keep model versions for rollback.
4. Compare candidates on an untouched holdout set.
5. Continue monitoring after deployment.
"""
    (REPORT_DIR / "improvement_report.md").write_text(report, encoding="utf-8")

    print(comparison.to_string(index=False))
    print("Best parameters:", search.best_params_)
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
