# Week 8 - Task 3: Machine Learning Model Deployment API

## Overview

This project demonstrates how to move a trained machine-learning model into a locally served REST API using **FastAPI**.

The API:
- loads a serialized scikit-learn model at application startup,
- validates JSON requests,
- accepts exactly 30 numeric features,
- performs real-time inference,
- returns the predicted class and probabilities,
- exposes a health endpoint,
- and can be tested with cURL.

## Model

The included `app/model.joblib` is a trained Gradient Boosting classifier using the Scikit-Learn Breast Cancer Wisconsin (Diagnostic) dataset.

The model artifact is included so the API can be started immediately.

To regenerate it:

```bash
python app/train_model.py
```

## Project Structure

```text
Task 3/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py
│   ├── train_model.py
│   └── model.joblib
└── tests/
    └── test_api.sh
```

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Start the API

From the Task 3 root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### GET /health

Checks whether the model is loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Request body:

```json
{
  "features": [30 numeric values]
}
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "malignant",
  "probabilities": {
    "malignant": 0.91,
    "benign": 0.09
  }
}
```

## Test with cURL

Linux/macOS/Git Bash:

```bash
bash tests/test_api.sh
```

Or use the Swagger UI at `/docs`.

For Windows PowerShell, an equivalent request is:

```powershell
$body = @{
  features = @(17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

## Error Handling

FastAPI/Pydantic rejects requests with fewer or more than 30 features.

The inference endpoint also returns HTTP 400 for invalid numeric values and HTTP 503 if the model has not loaded.

## Latency and Scalability

For this educational local deployment, model loading occurs once during startup rather than once per request. This reduces repeated disk I/O and keeps inference requests lightweight.

For production, consider:
- multiple Uvicorn/Gunicorn workers,
- containerization,
- request logging and monitoring,
- authentication,
- HTTPS,
- input/data drift checks,
- model versioning,
- and load testing.

This project demonstrates model serving concepts and is not intended as a medical diagnostic service.
