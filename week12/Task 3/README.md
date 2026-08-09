# Week 12 Task 3 — Automated Model Monitoring and Retraining

## Objective
Monitor a deployed machine learning model, detect performance degradation/model drift, and trigger retraining when a configurable threshold is reached.

## Deliverables
- Monitoring and drift detection logic
- Automated retraining script
- Monitoring dashboard
- Performance report
- Documentation

## Project structure
```text
Task 3/
├── app/
│   ├── monitoring.py
│   ├── retrain.py
│   └── model.joblib
├── dashboard/
│   └── monitoring_dashboard.py
├── data/
│   └── new_data.csv
├── reports/
│   └── monitoring_report.md
├── README.md
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python app/monitoring.py
python app/retrain.py
python dashboard/monitoring_dashboard.py
```

The monitoring script calculates current model accuracy and a simple feature-distribution drift score. Retraining is triggered when the monitored accuracy falls below the configured threshold or drift exceeds the configured threshold.

## Workflow
```text
New production data
        ↓
Performance monitoring
        ↓
Drift / degradation detection
        ↓
Threshold exceeded?
   ↓ yes       ↓ no
Retrain       Continue monitoring
   ↓
Validate new model
   ↓
Save replacement model
```

## Production extension
In production, the monitoring job can run on a schedule or be triggered by fresh labeled data. Retraining should normally include validation, model versioning, approval gates, and rollback support before replacing a production model.
