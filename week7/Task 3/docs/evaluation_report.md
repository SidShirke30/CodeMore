# Model Evaluation & Interpretability Report

## 1. Objective

The goal of this task is to evaluate a classification model using multiple complementary metrics and explain its predictions using SHAP.

The model used is Gradient Boosting on the Breast Cancer Wisconsin (Diagnostic) dataset.

## 2. Evaluation Methodology

The data is split into training and test sets using stratification and a fixed random seed.

The following metrics are calculated on the test set:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

A confusion matrix provides a direct count of correct and incorrect classifications.

## 3. ROC-AUC

The ROC curve measures the relationship between true-positive rate and false-positive rate across classification thresholds.

AUC summarizes this ranking behavior. It is threshold-independent, but it should not replace precision, recall, F1, and confusion-matrix analysis.

## 4. SHAP Explainability

SHAP assigns contribution values to features for individual predictions.

### Global interpretation

The SHAP summary plot identifies features with large average absolute contributions. The color distribution indicates whether high or low feature values tend to contribute to the model output in a particular direction.

### Local interpretation

The waterfall plot and force plot explain one specific test prediction. Features with positive contributions push the model output toward one class, while negative contributions push it toward the other relative to the model's baseline.

## 5. Domain Interpretation

The dataset contains measurements derived from images of breast masses. Features associated with cell size, shape, texture, concavity, and related geometry can therefore have meaningful predictive signal.

However, SHAP describes **model behavior**, not biological causation. A high SHAP contribution means the feature influenced the trained model's prediction; it does not establish that the feature causes the underlying condition.

## 6. Generated Results

After running the notebook, the following files contain the actual measured results:

```text
assets/evaluation_metrics.csv
assets/shap_feature_importance.csv
assets/confusion_matrix.png
assets/roc_curve.png
assets/shap_summary.png
assets/shap_feature_importance.png
assets/shap_local_waterfall.png
assets/shap_force_plot.html
```

Use the generated metric values and plots when completing the final submission.

## 7. Conclusion

The experiment demonstrates that evaluating a classifier requires more than a single accuracy value. Precision, recall, F1-score, confusion matrices, and ROC-AUC reveal different aspects of predictive behavior.

SHAP adds an interpretability layer by showing which features influenced predictions globally and locally. Together, these tools provide a more complete understanding of model performance and behavior.
