"""Custom evaluation metrics for imbalanced binary classification."""

import numpy as np
import pandas as pd


def custom_roc_curve(y_true, y_score):
    """Calculate ROC points without sklearn.metrics.roc_curve."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)

    if positives == 0 or negatives == 0:
        raise ValueError("ROC curve requires both classes.")

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    # Keep points at distinct score thresholds.
    distinct = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    tp = tps[distinct]
    fp = fps[distinct]

    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    thresholds = np.r_[np.inf, scores[distinct]]

    return fpr, tpr, thresholds


def custom_precision_recall_curve(y_true, y_score):
    """Calculate precision-recall points without sklearn's PR helper."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)

    distinct = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]

    tp = tp[distinct]
    fp = fp[distinct]

    precision = tp / np.maximum(tp + fp, 1)
    positives = np.sum(labels == 1)
    recall = tp / max(positives, 1)

    # Include the conventional starting recall/precision point.
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = scores[distinct]

    return precision, recall, thresholds


def f1_scores_by_average(y_true, y_pred):
    """Return macro, micro, and weighted F1 for binary/multiclass labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)

    per_class = []
    supports = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class.append(f1)
        supports.append(np.sum(y_true == cls))

    per_class = np.asarray(per_class)
    supports = np.asarray(supports)

    macro = np.mean(per_class)
    micro_tp = np.sum(y_true == y_pred)
    # For single-label classification, micro precision/recall/F1 equals accuracy.
    micro = micro_tp / len(y_true)
    weighted = np.average(per_class, weights=supports)

    return {
        "macro": float(macro),
        "micro": float(micro),
        "weighted": float(weighted),
    }


def threshold_metrics(y_true, y_score, thresholds):
    """Evaluate precision, recall, F1 and confusion counts by threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    rows = []
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })

    return pd.DataFrame(rows)
