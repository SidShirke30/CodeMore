"""Random Search hyperparameter optimization pipeline."""
import time
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RANDOM_STATE = 42

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    param_distributions = {
        "n_estimators": np.arange(50, 301, 25),
        "learning_rate": np.linspace(0.01, 0.20, 20),
        "max_depth": np.arange(1, 5),
        "min_samples_split": np.arange(2, 11),
        "min_samples_leaf": np.arange(1, 8),
        "subsample": np.linspace(0.7, 1.0, 7),
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    pred = search.best_estimator_.predict(X_test)
    prob = search.best_estimator_.predict_proba(X_test)[:, 1]

    result = {
        "method": "Random Search",
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "execution_seconds": elapsed,
        "n_iter": 30,
    }

    print(json.dumps(result, indent=2, default=str))
    return result, search.cv_results_

if __name__ == "__main__":
    main()
