# Week 8 - Task 2: Advanced Hyperparameter Optimization

## Overview

This project compares Random Search and Bayesian Optimization for tuning a Gradient Boosting classifier.

### Methods
- Baseline model
- RandomizedSearchCV
- Optuna TPE Bayesian Optimization
- 5-fold cross-validation
- ROC-AUC optimization
- Test-set accuracy, F1, ROC-AUC
- Execution-time comparison

## Structure

```text
Task 2/
├── README.md
├── requirements.txt
├── src/
│   ├── random_search.py
│   └── bayesian_opt.py
└── docs/
    └── optimization_report.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Random Search

```bash
python src/random_search.py
```

## Run Bayesian Optimization

```bash
python src/bayesian_opt.py
```

Both methods use the same dataset, train/test split, scoring metric, cross-validation strategy, and 30-evaluation budget to make the comparison fair.

## Comparison

Record the printed:
- best hyperparameters,
- best cross-validation ROC-AUC,
- test accuracy,
- test F1,
- test ROC-AUC,
- execution time.

Use these values to compare final performance and search efficiency.

## Note

Optuna's TPE sampler is a sequential model-based optimization strategy commonly used for Bayesian-style hyperparameter optimization.
