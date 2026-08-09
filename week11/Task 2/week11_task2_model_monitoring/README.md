# Week 11 Task 2 — Image Classification Model Monitoring

## Objective
Monitor a deployed image-classification model by logging prediction outcomes and tracking accuracy, precision, and recall over time. The project also demonstrates simple drift/model-decay detection.

## Project Structure
```text
week11_task2_model_monitoring/
├── README.md
├── requirements.txt
├── app/
│   ├── metrics_tracker.py
│   └── monitoring.py
├── dashboard/
│   └── performance_dashboard.py
├── logs/
│   └── performance.csv
└── reports/
    └── monitoring_report.md
```

## Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run monitoring simulation
```bash
python app/monitoring.py
```

This generates prediction/performance logs and checks for potential model decay.

## Generate dashboard
```bash
python dashboard/performance_dashboard.py
```

The dashboard saves a performance plot under `dashboard/performance_metrics.png`.

## Metrics
- Accuracy
- Precision
- Recall
- Error rate

## Drift / decay detection
The monitoring script compares recent performance against an earlier baseline. A warning is produced when accuracy drops beyond the configured threshold.

For a production system, the same tracker can be connected to the prediction endpoint from Week 11 Task 1 and populated with real labelled feedback.
