# Model Evaluation and Optimization Report

## 1. Objective
The objective is to establish a baseline Random Forest classifier, estimate its generalization performance using cross-validation, optimize its hyperparameters, and analyze whether the final model exhibits high bias or high variance.

## 2. Dataset
The Breast Cancer Wisconsin Diagnostic dataset from Scikit-Learn is used. It contains 569 observations, 30 numeric features, and a binary target.

A stratified 80/20 split is used so that the held-out test set is not used during hyperparameter selection.

## 3. Baseline Model
A Random Forest classifier with a reproducible configuration is trained first. Accuracy, precision, recall, F1-score, ROC-AUC, and training time provide the baseline.

## 4. Cross-Validation
Five-fold stratified cross-validation is used on the training data. Cross-validation gives a more robust estimate of model performance than relying on one training/validation split.

## 5. Hyperparameter Tuning
GridSearchCV evaluates a predefined search space containing parameters such as:
- number of trees,
- maximum tree depth,
- minimum samples required for splitting,
- minimum samples per leaf,
- feature selection strategy.

The optimization objective is ROC-AUC.

## 6. Learning Curves
Learning curves compare training and validation scores at increasing training-set sizes.

Interpretation:
- High training and low validation scores indicate high variance/overfitting.
- Both low training and validation scores indicate high bias/underfitting.
- Converging high scores indicate a good fit.

## 7. ROC Curve
The ROC curve plots true-positive rate against false-positive rate over classification thresholds. ROC-AUC summarizes ranking performance independently of one fixed threshold.

## 8. Final Evaluation
The optimized model is evaluated exactly once on the held-out test set after model selection.

The notebook produces a comparison table containing baseline and optimized metrics.

## 9. Performance Gains
The final report should use the notebook's generated values to describe:
- cross-validation improvement,
- test-set improvement,
- ROC-AUC improvement,
- training-time trade-offs,
- and changes in the bias/variance profile.

Exact numeric results should be taken from the executed notebook rather than assumed in advance.

## 10. Conclusion
The recommended model should balance predictive performance, generalization, training cost, and complexity. Hyperparameter tuning is useful when it produces a measurable improvement on unseen data without creating excessive variance.
