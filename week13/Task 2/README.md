# Week 13 Task 2: ML Model Monitoring and Alerts

A production-style monitoring project for tracking deployed ML model performance, latency, and data drift. The system records metrics and raises alerts when configurable thresholds are exceeded.

## Structure

```text
week13_task2_model_monitoring_alerts/
├── app/
│   ├── monitoring.py
│   └── alerting.py
├── dashboard/
│   └── monitoring_dashboard.py
├── data/
│   ├── baseline.csv
│   └── production_metrics.csv
├── reports/
│   └── monitoring_report.md
├── config/
│   └── monitoring_config.json
├── requirements.txt
└── README.md
```

## Run monitoring

```bash
pip install -r requirements.txt
python app/monitoring.py
```

The script calculates accuracy, average latency, and PSI-based data drift, then evaluates alert thresholds.

## Run dashboard

```bash
python dashboard/monitoring_dashboard.py
```

The dashboard creates a simple visualization of accuracy and latency over time.

## Alerts

Alerts are generated for:
- Accuracy below the configured threshold.
- Latency above the configured threshold.
- PSI drift above the configured threshold.

The configuration is stored in `config/monitoring_config.json`.
