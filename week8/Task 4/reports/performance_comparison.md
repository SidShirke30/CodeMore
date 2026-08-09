# Performance Comparison Report

## Objective
Compare a baseline Decision Tree with Random Forest and Gradient Boosting ensemble models.

## Dataset
The Breast Cancer Wisconsin Diagnostic dataset from Scikit-Learn contains 569 samples and 30 numeric features. A stratified 80/20 train/test split is used.

## Models
- **Decision Tree:** interpretable but can have high variance and overfit.
- **Random Forest:** averages many randomized trees to reduce variance.
- **Gradient Boosting:** sequentially builds trees that correct previous errors.

## Metrics
The notebook reports Accuracy, Precision, Recall, F1-score, ROC-AUC, training time, and the train/test accuracy gap.

## Interpretation
Use the generated results to select the best model based on test performance, overfitting gap, training time, and complexity. Do not assume a winner before executing the notebook because results can vary slightly with library versions.

## Strengths and weaknesses

| Model | Strengths | Weaknesses |
|---|---|---|
| Decision Tree | Simple and interpretable | Can overfit |
| Random Forest | Robust and reduces variance | Larger, less interpretable |
| Gradient Boosting | Strong predictive performance | More sensitive to hyperparameters |

Ensemble learning improves stability or predictive performance by combining multiple learners rather than relying on a single tree.
