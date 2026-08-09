# CI/CD Pipeline Documentation

## Tool
This project uses **Jenkins** as the CI/CD automation server.

## Workflow

```text
Git commit
   ↓
Jenkins checkout
   ↓
Install dependencies
   ↓
Train model
   ↓
Automated tests
   ↓
Build Docker image
   ↓
Deploy/run container
```

## Continuous Integration
Every pipeline run installs the declared dependencies, trains the model from the source dataset, and executes automated tests. A failed test stops the pipeline before deployment.

## Continuous Deployment
After successful tests, Jenkins builds a versioned Docker image and runs it as the deployment step. In a production environment, this stage can be replaced with deployment to a container registry and cloud service.

## Reliability practices
- Pin Python dependency versions.
- Keep training and tests reproducible with fixed random seeds.
- Fail the pipeline when tests fail.
- Version Docker images using the Jenkins build number.
- Keep deployment steps automated and repeatable.
- Store secrets in Jenkins credentials rather than source code.

## Jenkins prerequisites
The Jenkins agent should have:
- Python 3
- pip
- Docker
- Git
- Permission to execute Docker commands

## Production extension
A production pipeline could add model validation, security scanning, image publishing, staging deployment, approval gates, monitoring checks, and automatic rollback.
