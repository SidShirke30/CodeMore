# Week 9 Task 4 — AI Ethics, Bias & Fairness

This project explores ethical considerations in machine learning with a focus on bias detection, fairness metrics, and mitigation.

## Project goals
- Identify potential dataset and model bias.
- Evaluate classification performance across protected groups.
- Calculate fairness metrics such as demographic parity, selection rate, TPR, and FPR.
- Apply a fairness-aware mitigation strategy using group-aware threshold adjustment.
- Discuss ethical limitations and responsible deployment.

## Dataset
A synthetic hiring-style classification dataset is generated in the notebook so the project is fully reproducible without requiring an external download.

Protected attribute: `group` (A/B).

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/fairness_analysis.ipynb
```

The notebook generates the synthetic data, trains a baseline model, evaluates group-level fairness, applies mitigation, and compares results.
