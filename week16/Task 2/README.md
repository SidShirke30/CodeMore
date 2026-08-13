# Week 16 Task 2 — Cloud-Based ML REST API

## Objective
Deploy a simple machine learning model as a REST API suitable for a cloud platform.

## Model
A Logistic Regression classifier is trained on the Iris dataset using scikit-learn.

## API
- `GET /health` — health check
- `POST /predict` — accepts JSON feature values and returns a prediction

Example request:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Example response:
```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

## Run locally

```bash
pip install -r requirements.txt
python app/app.py
```

The API runs on `http://localhost:5000`.

## Docker

```bash
docker build -t week16-ml-api .
docker run -p 5000:5000 week16-ml-api
```

## Cloud deployment

The included Dockerfile can be deployed to a cloud container service. The application reads the `PORT` environment variable so a managed platform can provide its listening port.

A generic deployment flow is documented in `docs/cloud_deployment.md`.

## Project structure

```text
Task 2/
├── app/
│   ├── app.py
│   └── train_model.py
├── docs/
│   └── cloud_deployment.md
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```
