"""Fairness metrics and a simple fairness-aware threshold pipeline."""

import numpy as np
import pandas as pd


def group_metrics(y_true, y_pred, group):
    """Return selection rate, TPR, FPR and accuracy for each group."""
    rows = []
    df = pd.DataFrame({"y": y_true, "pred": y_pred, "group": group})

    for g, part in df.groupby("group"):
        tp = int(((part.y == 1) & (part.pred == 1)).sum())
        tn = int(((part.y == 0) & (part.pred == 0)).sum())
        fp = int(((part.y == 0) & (part.pred == 1)).sum())
        fn = int(((part.y == 1) & (part.pred == 0)).sum())

        rows.append({
            "group": g,
            "selection_rate": part.pred.mean(),
            "tpr": tp / (tp + fn) if tp + fn else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / len(part),
        })

    return pd.DataFrame(rows)


def demographic_parity_difference(metrics):
    """Difference between maximum and minimum group selection rates."""
    return float(metrics["selection_rate"].max() - metrics["selection_rate"].min())


def equal_opportunity_difference(metrics):
    """Difference between maximum and minimum group TPR."""
    return float(metrics["tpr"].max() - metrics["tpr"].min())


def group_threshold_predictions(probabilities, groups, thresholds):
    """Apply a separate decision threshold per group."""
    probabilities = np.asarray(probabilities)
    groups = np.asarray(groups)
    return np.array([
        int(p >= thresholds[g]) for p, g in zip(probabilities, groups)
    ])
