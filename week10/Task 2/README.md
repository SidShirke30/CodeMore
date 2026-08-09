# Week 10 Task 2 — ML Model Monitoring & Drift Detection

This project adds production-style monitoring to a deployed Flask machine-learning model.

## Features
- Logs input features, predictions, latency, and optional actual labels.
- Tracks request count, prediction distribution, latency, and accuracy.
- Provides `/metrics` for current monitoring metrics.
- Includes a PSI-style prediction-distribution drift calculation.
- Generates a markdown monitoring report.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app/train_model.py
python app/app.py
```

The API runs at `http://127.0.0.1:5000`.

## Endpoints
- `GET /health`
- `POST /predict`
- `GET /metrics`

Example:
```json
{"features": [5.1, 3.5, 1.4, 0.2]}
```

For delayed performance measurement:
```json
{"features": [5.1, 3.5, 1.4, 0.2], "actual": 0}
```

Prediction records are stored in `logs/predictions.csv`.

## Generate the report
```powershell
python dashboard/performance_dashboard.py
```

The demonstration model uses the scikit-learn Iris dataset. Production systems should additionally use centralized telemetry, authentication, privacy controls, alerting, persistent metrics storage, and validated drift thresholds.
