# Week 10 Task 1 — Flask ML Model Integration

## Overview
This project integrates a trained scikit-learn machine learning model into a modular Flask web application. The API accepts JSON feature values and returns a real-time prediction.

## Project Structure
```text
week10/
└── Task 1/
    ├── README.md
    ├── requirements.txt
    ├── app/
    │   ├── app.py
    │   ├── model.pkl
    │   └── train_model.py
    └── docs/
        └── api_usage.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If the model file is missing, generate it:

```powershell
python app/train_model.py
```

Start Flask:

```powershell
python app/app.py
```

The API runs at `http://127.0.0.1:5000`.

## API
- `GET /health` — health check
- `POST /predict` — accepts JSON features and returns a prediction

See `docs/api_usage.md` for request examples.

## Example request
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## Example response
```json
{
  "prediction": 0,
  "probabilities": [0.98, 0.02, 0.0]
}
```

## Notes
The example model uses the Iris dataset and a Random Forest classifier so the repository is fully runnable without downloading external data.
