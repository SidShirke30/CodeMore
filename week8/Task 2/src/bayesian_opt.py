"""Optuna Bayesian-style hyperparameter optimization pipeline."""
import time
import json
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RANDOM_STATE = 42

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "random_state": RANDOM_STATE,
        }
        model = GradientBoostingClassifier(**params)
        return cross_val_score(
            model, X_train, y_train, scoring="roc_auc", cv=5, n_jobs=-1
        ).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    start = time.perf_counter()
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    elapsed = time.perf_counter() - start

    best_params = dict(study.best_params)
    best_params["random_state"] = RANDOM_STATE
    best_model = GradientBoostingClassifier(**best_params)
    best_model.fit(X_train, y_train)

    pred = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:, 1]

    result = {
        "method": "Optuna TPE Bayesian Optimization",
        "best_params": best_params,
        "cv_best_score": float(study.best_value),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "execution_seconds": elapsed,
        "n_trials": 30,
    }

    print(json.dumps(result, indent=2, default=str))
    return result, study

if __name__ == "__main__":
    main()
