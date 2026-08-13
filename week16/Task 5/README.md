# Week 16 Task 5 — A/B Testing for AI Model Versions

This project implements a simple A/B testing framework for comparing two deployed model versions.

## Deliverables
- A/B testing framework with configurable traffic split
- Evaluation metrics: accuracy, latency, and error rate
- Statistical comparison using a two-proportion z-test for accuracy
- Experiment report with model-selection recommendation
- Example data and tests
- Dashboard script for visual comparison

## Run
```bash
pip install -r requirements.txt
python app/ab_test.py
python app/dashboard.py
pytest
```
