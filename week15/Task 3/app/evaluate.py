def classification_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Inputs must have equal non-zero length")

    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    correct = sum(a == b for a, b in zip(y_true, y_pred))

    accuracy = correct / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def candidate_is_better(current, candidate, min_f1_improvement=0.01):
    return (
        candidate["accuracy"] >= current["accuracy"]
        and candidate["f1"] >= current["f1"] + min_f1_improvement
    )
