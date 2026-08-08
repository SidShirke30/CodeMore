# Metric Suitability Report: Imbalanced Classification

## 1. Business Context

The project models a fraud-detection scenario. Fraudulent transactions are rare compared with legitimate transactions.

In such a setting, a classifier can achieve high accuracy simply by predicting the majority class most of the time. Therefore, evaluation must explicitly measure minority-class performance.

## 2. Baseline Findings

The baseline Logistic Regression model is intentionally trained without imbalance handling. The notebook calculates accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.

The actual values are generated when the notebook is executed and saved to:

```text
docs/assets/evaluation_summary.csv
```

## 3. ROC Curve and ROC-AUC

ROC analysis compares true-positive rate with false-positive rate over many thresholds.

ROC-AUC is useful for assessing ranking performance across thresholds. However, under severe class imbalance, the false-positive rate can remain numerically small even when the number of false positives is operationally significant.

Therefore ROC-AUC should not be the only metric.

## 4. Precision-Recall Curve and PR-AUC

The Precision-Recall curve directly focuses on positive-class detection.

For rare fraud:
- Recall measures how much fraud is captured.
- Precision measures how many alerts are actually fraud.

PR-AUC / average precision is often more informative than ROC-AUC when the positive class is very rare because it emphasizes performance on positive predictions.

## 5. Threshold Trade-Off

Changing the classification threshold changes precision and recall.

A lower threshold generally catches more positive cases, increasing recall, but can create more false positives and reduce precision.

A higher threshold generally increases precision but can miss more positives.

The appropriate threshold should be selected using business costs, investigation capacity, and acceptable fraud-loss risk.

## 6. Macro, Micro, and Weighted F1

### Macro F1

Macro F1 gives each class equal importance. It is useful when minority-class performance should not be hidden by the majority class.

### Micro F1

Micro F1 aggregates decisions across classes. In single-label classification it is closely related to accuracy and can therefore be dominated by the majority class.

### Weighted F1

Weighted F1 weights each class by its number of observations. It reflects overall population performance but can still give the majority class substantial influence.

For severe imbalance, macro F1 is useful as a fairness-to-classes diagnostic, while positive-class recall, precision, F1, and PR-AUC should be monitored for the business objective.

## 7. Recommended Metric

For the simulated fraud-detection scenario, **PR-AUC together with positive-class recall and precision** is the recommended evaluation combination.

If missing fraud is extremely expensive, recall should receive greater weight and the operating threshold can be lowered.

If investigation capacity is limited, precision becomes more important.

F1 is appropriate when precision and recall have approximately equal importance.

Accuracy should be treated as a secondary metric rather than the primary decision metric.

## 8. Conclusion

Imbalanced classification requires metrics that reveal minority-class behavior. ROC-AUC is valuable for general ranking analysis, but Precision-Recall analysis is particularly useful when positive events are rare.

The notebook's custom metric implementations, threshold analysis, and macro/micro/weighted F1 comparison provide a reproducible framework for selecting metrics based on the actual business cost of classification errors.
