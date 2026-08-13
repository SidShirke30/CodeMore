# Week 14 Task 2 — Model Monitoring and Logging

## Objective
Monitor a deployed Flask ML API over time by logging API requests and predictions, calculating performance metrics, and visualizing model performance.

## Features
- Request and prediction logging
- Accuracy, precision, recall, and latency tracking
- CSV-based monitoring history
- Performance dashboard
- Basic degradation checks
- Unit tests for monitoring utilities

## Run
```bash
pip install -r requirements.txt
python app/monitoring.py
python dashboard/performance_dashboard.py
pytest
```

See `reports/monitoring_report.md` for the initial performance analysis.
