# Week 12 Task 1 — Dockerized ML Model Deployment

## Objective
Package a machine learning model and its Python dependencies into a Docker container and document a cloud deployment strategy.

## Project structure
```text
Task 1/
├── app/
│   ├── app.py
│   ├── train_model.py
│   └── model.joblib
├── docs/
│   └── deployment.md
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Run locally without Docker
```bash
pip install -r requirements.txt
python app/train_model.py
python app/app.py
```

API:
- `GET /health`
- `POST /predict`

Example request:
```json
{"features": [5.1, 3.5, 1.4, 0.2]}
```

## Run with Docker
Build:
```bash
docker build -t week12-ml-api .
```

Run:
```bash
docker run --rm -p 5000:5000 week12-ml-api
```

Test:
```bash
curl http://localhost:5000/health
```

PowerShell:
```powershell
Invoke-RestMethod -Uri http://localhost:5000/health
```

## Learning outcomes
- Understand containerization for ML workloads.
- Package application code and dependencies consistently.
- Expose a prediction API from a Docker container.
- Understand portability, reproducibility and scalability.
- Review a cloud deployment approach.
