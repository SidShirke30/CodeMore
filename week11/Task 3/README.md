# Week 11 Task 3 — Model Performance Improvement

## Objective
Improve or maintain model performance after monitoring reveals degradation. This project demonstrates retraining with newer data, hyperparameter tuning, and model comparison.

## Workflow
1. Train a baseline Random Forest.
2. Simulate newly available labelled production data.
3. Retrain using the updated data.
4. Tune hyperparameters with GridSearchCV.
5. Compare baseline and improved models.
6. Save the recommended model and report.

## Run
```bash
pip install -r requirements.txt
python app/improve_model.py
```

Outputs:
- `data/model_comparison.csv`
- `reports/improvement_report.md`
- `app/improved_model.joblib`

## Maintenance
Retrain only after validating that degradation is real and new data is representative. Keep model versions and validate candidates on an untouched holdout set.
