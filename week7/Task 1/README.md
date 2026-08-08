# Week 7 - Task 1: Advanced Feature Engineering

## Overview

This project demonstrates advanced feature engineering on the **UCI Wine Quality - Red** tabular dataset.

The notebook compares a baseline Linear Regression model with models using:

- Polynomial features
- Pairwise interaction terms
- Logarithmic transformations for skewed positive variables
- A combined feature-engineering approach

The objective is to measure how engineered features affect predictive performance and interpretability.

## Project Structure

```text
Task 1/
├── README.md
├── requirements.txt
├── notebooks/
│   └── feature_engineering.ipynb
├── docs/
│   └── impact_report.md
└── data/
    └── raw_dataset_info.txt
```

Additional files such as plots and metric CSV/JSON files are generated when the notebook is executed.

## Dataset

**UCI Wine Quality - Red**

Source: UCI Machine Learning Repository

The dataset contains physicochemical measurements for red wine and an integer `quality` score. The notebook downloads the dataset automatically from UCI if it is not already present in `data/`.

## How to Run

From the Task 1 directory:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open:

```text
notebooks/feature_engineering.ipynb
```

Run all cells from top to bottom.

If the environment does not have Jupyter installed, install it with:

```bash
pip install notebook
```

## Expected Outputs

Running the notebook creates:

```text
docs/
├── actual_vs_predicted.png
├── log_feature_distributions.png
├── raw_feature_distributions.png
├── performance_comparison.csv
└── metrics_summary.json
```

These outputs support the written impact report.

## Key Design Choices

### Baseline
Standardized Linear Regression provides a simple benchmark.

### Polynomial + Interactions
Degree-2 PolynomialFeatures exposes squared and pairwise interaction relationships. Ridge regularization helps control multicollinearity.

### Mathematical Transformations
`log1p` is applied to selected non-negative, potentially skewed features.

### Evaluation
The same held-out test set is used for every model so performance comparisons are fair.

## Reproducibility

The train/test split uses `random_state=42`.

