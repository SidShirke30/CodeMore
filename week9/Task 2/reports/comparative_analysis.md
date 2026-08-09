# Comparative Analysis - Advanced Ensemble Learning

## Objective
The goal is to compare Random Forest, Gradient Boosting, and AdaBoost while examining predictive performance, computational cost, and bias/variance trade-offs.

## Dataset
The Breast Cancer Wisconsin Diagnostic dataset from Scikit-Learn is used for binary classification. A stratified 80/20 train/test split is used, with five-fold stratified cross-validation performed only on the training set.

## Methods

### Random Forest
Random Forest uses bagging and randomized feature selection to train many decision trees independently and aggregate their predictions. It is generally robust and effective at reducing variance.

### Gradient Boosting
Gradient Boosting builds weak learners sequentially, with each learner attempting to correct errors made by previous learners. It can achieve excellent predictive performance but is more sensitive to hyperparameters.

### AdaBoost
AdaBoost sequentially reweights difficult observations so later weak learners focus more strongly on previously misclassified examples. It can be powerful on clean datasets but may be sensitive to noisy observations and outliers.

## Hyperparameter Tuning
The code uses GridSearchCV with five-fold stratified cross-validation. The search spaces are stored in `config/hyperparameters.json`.

The optimization metric is F1-score because it balances precision and recall for the classification problem.

## Evaluation
The script reports:
- Cross-validation F1
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training accuracy
- Train/test accuracy gap
- Search and training time
- Best hyperparameters

Run the script to generate `reports/model_results.csv`.

## Bias and Variance
A large train/test gap can indicate high variance or overfitting. If both training and test performance are low, the model may have high bias. Ensemble methods can reduce these problems in different ways: Random Forest primarily reduces variance through averaging, while boosting methods progressively reduce bias.

## Trade-offs

| Method | Main advantage | Main drawback |
|---|---|---|
| Random Forest | Robust, parallelizable, strong variance reduction | Can use substantial memory and is less interpretable |
| Gradient Boosting | Often excellent predictive performance | Sensitive to learning rate, depth and number of estimators |
| AdaBoost | Simple and effective boosting strategy | Can be affected by noisy or mislabeled observations |

## Final Model Selection
Select the final model using held-out test performance, cross-validation stability, computational cost, and the train/test gap. Do not assume one algorithm always wins; use the generated `model_results.csv` values to support the conclusion.
