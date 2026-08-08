# Week 8 - Task 1: Evaluation Metrics for Imbalanced Datasets

## Overview

This project explores evaluation metrics for a highly imbalanced classification problem using a simulated fraud-detection dataset.

It demonstrates:
- baseline classification,
- custom ROC curve calculation,
- custom Precision-Recall curve calculation,
- ROC-AUC and PR-AUC,
- precision/recall/F1 threshold analysis,
- macro, micro, and weighted F1,
- and business-oriented metric selection.

## Project Structure

```text
Task 1/
├── README.md
├── requirements.txt
├── notebooks/
│   └── imbalanced_evaluation.ipynb
├── docs/
│   └── metric_suitability_report.md
└── src/
    └── custom_metrics.py
```

## Dataset and Business Context

The notebook generates a reproducible synthetic dataset with approximately 1.5% positive cases. The positive class represents potentially fraudulent transactions.

This setup demonstrates why accuracy can be misleading when positive events are rare.

## Run

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
notebooks/imbalanced_evaluation.ipynb
```

Run all cells from top to bottom.

## Evaluation

The notebook produces:
- ROC curve
- Precision-Recall curve
- threshold trade-off plot
- macro/micro/weighted F1 comparison
- evaluation summary
- threshold analysis CSV

Generated files are written under `docs/assets/`.

## Metric Guidance

For severe imbalance, use a combination of recall, precision, F1, and PR-AUC rather than relying on accuracy alone.

The appropriate operating threshold depends on the relative cost of false positives and false negatives.
