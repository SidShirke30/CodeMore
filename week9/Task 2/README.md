# Week 9 Task 2 - Advanced Ensemble Learning

## Overview
Compare three ensemble approaches on a real-world classification dataset:
- Random Forest
- Gradient Boosting
- AdaBoost

The project includes preprocessing, consistent train/test evaluation, hyperparameter configurations, training-time measurement, and a comparative report.

## Dataset
The Scikit-Learn Breast Cancer Wisconsin Diagnostic dataset is used. It is bundled with Scikit-Learn, so no external download is required.

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
python src/ensemble_models.py
```

The script prints a comparison table and saves `reports/model_results.csv`.

## Structure
```text
Task 2/
├── README.md
├── requirements.txt
├── src/
│   └── ensemble_models.py
├── reports/
│   └── comparative_analysis.md
└── config/
    └── hyperparameters.json
```
