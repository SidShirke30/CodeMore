"""Train and tune Random Forest, Gradient Boosting, and SVM models.

Run from the Task 2 directory:
    python src/train_tuned_model.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC_AUC": roc_auc_score(y, prob),
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs = os.path.join(root, "docs")
    os.makedirs(docs, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=RANDOM_STATE
    )

    searches = []

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    searches.append(("Random Forest", GridSearchCV(
        rf, rf_grid, scoring="f1", cv=5, n_jobs=-1
    )))

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_dist = {
        "n_estimators": randint(50, 401),
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(1, 6),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 6),
        "subsample": uniform(0.7, 0.3),
    }
    searches.append(("Gradient Boosting", RandomizedSearchCV(
        gb, gb_dist, n_iter=30, scoring="f1", cv=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )))

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=RANDOM_STATE)),
    ])
    svm_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "model__kernel": ["rbf", "linear"],
    }
    searches.append(("SVM", GridSearchCV(
        svm, svm_grid, scoring="f1", cv=5, n_jobs=-1
    )))

    rows = []
    for name, search in searches:
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        result = metrics(search.best_estimator_, X_test, y_test)
        result["Model"] = name
        result["CV_F1"] = search.best_score_
        result["Search_Time_sec"] = elapsed
        result["Best_Params"] = str(search.best_params_)
        rows.append(result)

        print(f"\n{name}")
        print("Best CV F1:", search.best_score_)
        print("Test metrics:", {k: v for k, v in result.items()
                                if k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]})
        print("Best parameters:", search.best_params_)

    table = pd.DataFrame(rows).set_index("Model")
    table.to_csv(os.path.join(docs, "final_model_comparison.csv"))

    best = table.sort_values(
        ["F1", "ROC_AUC", "Search_Time_sec"],
        ascending=[False, False, True]
    ).index[0]

    with open(os.path.join(docs, "best_model.txt"), "w", encoding="utf-8") as f:
        f.write(f"Selected model: {best}\n")
        f.write("Selection priority: test F1, then ROC-AUC, then shorter search time.\n")

    print("\nSelected model:", best)
    print("Results saved to docs/final_model_comparison.csv")


if __name__ == "__main__":
    main()
