import json
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "hyperparameters.json"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

with CONFIG.open() as f:
    params = json.load(f)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        params["RandomForest"],
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        params["GradientBoosting"],
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        params["AdaBoost"],
    ),
}

results = []

for name, (model, grid) in models.items():
    start = time.perf_counter()

    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start
    best = search.best_estimator_

    train_pred = best.predict(X_train)
    test_pred = best.predict(X_test)
    test_prob = best.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "CV F1": search.best_score_,
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1": f1_score(y_test, test_pred),
        "ROC-AUC": roc_auc_score(y_test, test_prob),
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Train-Test Accuracy Gap": accuracy_score(y_train, train_pred)
            - accuracy_score(y_test, test_pred),
        "Training + Search Time (s)": elapsed,
        "Best Parameters": str(search.best_params_),
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nAdvanced Ensemble Comparison\n")
print(results_df.to_string(index=False))
results_df.to_csv(REPORTS / "model_results.csv", index=False)

print("\nResults saved to reports/model_results.csv")
