# Comparative Analysis Report

## Objective

The purpose of this experiment is to compare Random Forest, Gradient Boosting, and SVM classifiers and investigate whether systematic hyperparameter optimization improves their performance.

Both exhaustive grid search and randomized search are demonstrated.

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset from Scikit-Learn is used. It is a binary classification problem with 569 samples and 30 numerical features.

## Experimental Design

The data is divided into training, validation, and test sets.

The training portion is used by 5-fold cross-validation during hyperparameter search. The validation set is used for an additional model comparison, while the test set is held out until final evaluation.

## Search Methods

### GridSearchCV

GridSearchCV evaluates every combination in the supplied parameter grid. It is appropriate when the search space is manageable and an exhaustive comparison is desirable.

It is used for:
- Random Forest
- SVM

### RandomizedSearchCV

RandomizedSearchCV samples a fixed number of parameter combinations from probability distributions. It can explore larger spaces with a controlled computational budget.

It is used for:
- Gradient Boosting

## Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Search time

F1 is used as the primary cross-validation scoring metric because it balances precision and recall.

## Results

Run the notebook to generate:

```text
docs/final_model_comparison.csv
docs/best_model.txt
docs/best_model_confusion_matrix.png
```

The notebook prints the actual measured metrics. The report should be finalized using those generated values rather than hard-coded or assumed results.

## Model Selection

The final model should be selected by considering:

1. F1 performance.
2. ROC-AUC.
3. Generalization on the untouched test set.
4. Hyperparameter search time.
5. Model complexity and maintainability.

The best-performing model is not automatically the best practical model if its performance improvement is negligible compared with a simpler or substantially faster alternative.

## Conclusion

This project demonstrates a reproducible approach to model selection and hyperparameter optimization. GridSearchCV provides exhaustive optimization for manageable parameter spaces, while RandomizedSearchCV offers an efficient alternative for larger spaces.

The final notebook and generated comparison files provide the evidence needed to justify the chosen model.
