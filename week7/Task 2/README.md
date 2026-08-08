# Week 7 - Task 2: Model Selection & Hyperparameter Optimization

## Overview

This project compares three classification algorithms and tunes them with Scikit-Learn search utilities:

- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

The notebook demonstrates both `GridSearchCV` and `RandomizedSearchCV`, evaluates tuned models on a held-out test set, and produces a comparative analysis.

## Dataset

The project uses Scikit-Learn's built-in **Breast Cancer Wisconsin (Diagnostic)** dataset. It contains 569 samples, 30 numerical features, and a binary target.

Using a built-in dataset makes the project reproducible without a separate download.

## Project Structure

```text
Task 2/
├── README.md
├── requirements.txt
├── notebooks/
│   └── hyperparameter_tuning.ipynb
├── docs/
│   └── comparative_analysis.md
├── src/
│   └── train_tuned_model.py
└── data/
```

## Algorithms

### Random Forest
Tuned with `GridSearchCV` across tree count, depth, split size, leaf size, and feature-selection strategy.

### Gradient Boosting
Tuned with `RandomizedSearchCV` across estimator count, learning rate, depth, split size, leaf size, and subsampling.

### SVM
Tuned with `GridSearchCV` over `C`, `gamma`, and kernel. Standardization is included in a pipeline.

## Running the Notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/hyperparameter_tuning.ipynb
```

Run all cells from top to bottom.

## Running the Training Script

From the Task 2 directory:

```bash
python src/train_tuned_model.py
```

The script performs the same tuning workflow and saves final comparison files under `docs/`.

## Evaluation

Models are compared using:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Search/training time
- Model complexity

F1 is used as the main cross-validation optimization metric.

## Reproducibility

`random_state=42` is used wherever supported.
