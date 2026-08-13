# Week 14 Task 1 — Flask ML Prediction API

A Flask REST API that loads a trained machine learning model, accepts JSON input, and returns predictions.

## Features
- Flask prediction endpoint
- JSON request validation
- Model loading with joblib
- Health-check endpoint
- Unit tests using Flask's test client
- API usage documentation

## Structure
```text
Task 1/
├── app/
│   ├── app.py
│   ├── model.joblib
│   └── train_model.py
├── tests/
│   └── test_api.py
├── docs/
│   └── api_usage.md
├── README.md
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python app/app.py
```

API: `http://127.0.0.1:5000`

## Prediction
POST `/predict`

```json
{"features": [5.1, 3.5, 1.4, 0.2]}
```

## Health Check
GET `/health`

## Tests
```bash
pytest
```

The example model uses the Iris classification dataset.
