# Week 9 Task 3 - Flask ML Model Deployment

## Overview
This project demonstrates production-like serving of a trained machine learning model using Flask.

The application:
1. Trains a classification model.
2. Serializes it as `model.pkl`.
3. Loads the model once when Flask starts.
4. Exposes a `POST /predict` endpoint.
5. Validates incoming JSON.
6. Returns a prediction and probability.
7. Documents local testing and deployment challenges.

## Project Structure
```text
Task 3/
├── README.md
├── app.py
├── model.pkl
├── requirements.txt
└── docs/
    └── deployment_challenges.md
```

## Setup
Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need to regenerate the model:

```powershell
python app.py --train
```

Otherwise, start the API:

```powershell
python app.py
```

The server runs at:

```text
http://127.0.0.1:5000
```

## API Endpoint

### POST `/predict`

JSON body:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Example PowerShell request:

```powershell
$body = '{"features":[5.1,3.5,1.4,0.2]}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body $body
```

Example response:

```json
{
  "prediction": 0,
  "probability": 1.0
}
```

The model is trained on the Iris dataset and expects exactly four numeric features.

## Health Check

```text
GET /health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Input Validation
The API rejects:
- missing JSON
- missing `features`
- non-list feature values
- incorrect feature count
- non-numeric feature values

## Production Notes
This Flask development server is intended for local learning. Production deployments should use a WSGI server such as Gunicorn or Waitress, pin dependencies, configure logging, use environment-specific configuration, and add authentication/rate limiting where appropriate.
