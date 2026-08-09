# Week 9 Task 1: Model Evaluation, Cross-Validation & Hyperparameter Tuning

## Overview
This project demonstrates systematic model evaluation and optimization using a Random Forest classifier.

The workflow includes:
- K-fold cross-validation
- GridSearchCV hyperparameter tuning
- Learning curves for bias/variance analysis
- ROC curve and ROC-AUC
- Held-out test-set evaluation
- Comparison of baseline and optimized models
- A comprehensive evaluation report

## Structure
```text
Task 1/
├── README.md
├── requirements.txt
├── notebooks/
│   └── model_evaluation_tuning.ipynb
├── reports/
│   └── evaluation_report.md
└── plots/
    ├── learning_curves.png
    └── roc_curve.png
```

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Open the notebook and run all cells.

## Dataset
The notebook uses the Breast Cancer Wisconsin Diagnostic dataset bundled with Scikit-Learn, so no external download is required.

## Evaluation
The baseline Random Forest is compared with the GridSearchCV-optimized model using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-validation score
- Training time

Learning curves are used to diagnose high bias or high variance, while the ROC curve evaluates threshold-independent ranking performance.

## Reproducibility
A fixed random seed of 42 is used throughout the experiment.
