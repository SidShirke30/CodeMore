# Week 7 - Task 3: Model Evaluation & Interpretability

## Overview

This project evaluates a Gradient Boosting classification model using standard classification metrics and explains its predictions with SHAP.

The project demonstrates:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve
- ROC-AUC
- SHAP global explanations
- SHAP local explanations
- Feature-importance analysis

## Dataset

The project uses the Scikit-Learn **Breast Cancer Wisconsin (Diagnostic)** dataset.

It contains 569 observations, 30 numerical features, and a binary target.

This is an educational machine-learning exercise. The model should not be treated as a medical diagnostic tool.

## Project Structure

```text
Task 3/
├── README.md
├── requirements.txt
├── notebooks/
│   └── model_interpretation.ipynb
├── docs/
│   └── evaluation_report.md
└── assets/
    └── (generated visualizations)
```

## Installation

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/model_interpretation.ipynb
```

Run the notebook from top to bottom.

## Expected Assets

After execution, the notebook creates:

```text
assets/
├── confusion_matrix.png
├── roc_curve.png
├── shap_summary.png
├── shap_feature_importance.png
├── shap_local_waterfall.png
├── shap_force_plot.html
├── shap_feature_importance.csv
└── evaluation_metrics.csv
```

## Interpretation

Classification metrics quantify predictive performance, while SHAP explains how features contribute to individual model outputs.

SHAP feature importance should be interpreted as model attribution, not causal evidence.

## Reproducibility

The train/test split and model use `random_state=42`.
