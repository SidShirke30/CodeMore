# Week 10 Task 3 — A/B Testing for ML Models

A production-style Flask framework for comparing two machine-learning model variants using traffic splitting and statistical performance tracking.

## What it does
- Randomly assigns prediction requests to Model A or Model B.
- Supports deterministic assignment using a request/user ID.
- Logs variant, prediction, optional actual label, latency, and correctness.
- Exposes experiment metrics through an API endpoint.
- Generates a comparison report with accuracy, sample counts, and latency.
- Includes a two-proportion z-test to provide a simple statistical comparison when labeled outcomes are available.

## Project structure
```text
README.md
requirements.txt
app/
  app.py
  ab_testing.py
  train_models.py
  model_a.pkl
  model_b.pkl
reports/
  ab_test_report.md
logs/
  ab_results.csv
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app/train_models.py
python app/app.py
```

The server runs at `http://127.0.0.1:5000`.

## API endpoints

### Health
`GET /health`

### Predict
`POST /predict`

Request:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2],
  "user_id": "user-123",
  "actual": 0
}
```

`user_id` is optional. If supplied, the same user is deterministically assigned to the same variant. `actual` is optional and should be provided when ground truth becomes available.

Response:
```json
{
  "variant": "A",
  "prediction": 0,
  "latency_ms": 1.2
}
```

### Metrics
`GET /metrics`

Returns per-variant request count, labeled sample count, accuracy, and average latency.

## Generate the report
After collecting labeled requests:
```powershell
python reports/generate_report.py
```

## Experiment guidance
Keep the traffic allocation fixed during the comparison, define the primary metric before starting, avoid changing both models and traffic allocation at the same time, and collect enough observations before making a rollout decision. Statistical significance should be interpreted together with business impact and latency.

This demonstration uses the Iris dataset and two Random Forest variants. In a real deployment, the experiment should also include authentication, privacy controls, centralized telemetry, persistent storage, experiment versioning, and alerting.
