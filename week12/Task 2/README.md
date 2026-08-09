# Week 12 Task 2 — CI/CD Pipeline for Machine Learning

## Objective
Build a Jenkins-based CI/CD pipeline that automates machine learning model training, testing, packaging, and deployment.

## Pipeline stages
1. Checkout source code
2. Install Python dependencies
3. Train the model
4. Run automated tests
5. Build a Docker image
6. Deploy/run the container

## Project structure
```text
Task 2/
├── app/
│   ├── train_model.py
│   └── model.joblib
├── tests/
│   └── test_model.py
├── docs/
│   └── cicd_pipeline.md
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
├── README.md
└── .gitignore
```

## Local setup
```bash
pip install -r requirements.txt
python app/train_model.py
pytest -q
```

## Docker
```bash
docker build -t week12-ml-cicd .
docker run --rm week12-ml-cicd
```

## Jenkins
Create a Jenkins Pipeline job and point it to this repository. Jenkins will discover and execute the `Jenkinsfile`.

The pipeline is designed to fail fast if dependency installation, model training, tests, or Docker image creation fails.

## Benefits
- Repeatable model training
- Automated testing before deployment
- Consistent packaging
- Faster and safer releases
- Clear audit trail through Git commits and Jenkins builds
