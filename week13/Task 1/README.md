# Week 13 Task 1: Advanced ML Deployment

This project demonstrates production-oriented deployment of a machine learning model using:
- Docker containerization
- A Flask prediction API
- A serverless-compatible handler
- Deployment and scaling documentation

## Project Structure

```text
week13_task1_advanced_ml_deployment/
├── app/
│   ├── app.py
│   ├── model.joblib
│   └── train_model.py
├── serverless/
│   └── handler.py
├── docs/
│   └── deployment_and_scaling.md
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
python app/app.py
```

The API runs on `http://localhost:5000`.

### Prediction request

```bash
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{"features":[5.1,3.5,1.4,0.2]}"
```

## Docker

Build:

```bash
docker build -t week13-ml-api .
```

Run:

```bash
docker run -p 5000:5000 week13-ml-api
```

## Serverless

`serverless/handler.py` exposes the same prediction logic in a serverless-friendly function shape. The documentation explains how this can be adapted to AWS Lambda, Azure Functions, or Google Cloud Functions.

## Learning Objectives

- Understand portability and reproducibility through containers.
- Understand serverless scaling and its trade-offs.
- Separate model inference from deployment infrastructure.
- Document production-readiness and scaling considerations.
