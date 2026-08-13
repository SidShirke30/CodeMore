# Week 15 Task 2 — AI Model Performance Monitoring

## Objective
Monitor a deployed AI model by tracking accuracy, latency, resource utilization,
detecting performance degradation, and triggering alerts.

## Metrics
- Accuracy
- Average and percentile latency
- CPU and memory utilization
- Error rate
- Anomaly score

## Architecture
```text
Production Model
      |
      v
Metrics Collector
      |
      +----> Accuracy / Error Rate
      +----> Latency
      +----> CPU / Memory
      |
      v
Monitoring Store
      |
      v
Anomaly Detection
      |
      +----> Normal ----> Dashboard
      |
      +----> Degradation ----> Alert
```

## Components
- `app/metrics_collector.py` collects and calculates model metrics.
- `app/anomaly_detector.py` identifies abnormal metric values.
- `app/alerting.py` triggers alerts when thresholds are exceeded.
- `dashboard/monitoring_dashboard.py` creates a dashboard from recorded metrics.
- `reports/performance_report.md` documents the analysis.
- `config/monitoring_config.json` contains monitoring thresholds.

Run the dashboard with:
```bash
python dashboard/monitoring_dashboard.py
```

Run the monitoring check with:
```bash
python app/monitoring.py
```
