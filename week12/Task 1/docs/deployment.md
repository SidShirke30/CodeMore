# Deployment Documentation

## Why Docker?
Docker packages the application, Python runtime, libraries and supporting files into a portable image. This reduces environment-related differences between development and deployment.

## Container workflow
1. Build the image with `docker build`.
2. Run the container and expose port 5000.
3. Verify `/health`.
4. Send prediction requests to `/predict`.
5. Publish the image to a container registry.
6. Deploy the image to a cloud container service.

## Cloud strategy
The container can be deployed to AWS, Azure or GCP using a managed container service. A typical production flow is:

```text
Source code
   ↓
Docker build
   ↓
Container registry
   ↓
Managed cloud container service
   ↓
Load balancer / API
   ↓
Prediction clients
```

Examples of suitable managed services include AWS App Runner/ECS, Azure Container Apps, or Google Cloud Run.

## Scalability
Container replicas can be increased when request volume rises. Stateless prediction endpoints make horizontal scaling straightforward.

## Production considerations
- Pin dependency versions.
- Add authentication and HTTPS before exposing the API publicly.
- Store secrets outside the image.
- Add structured logging and health checks.
- Scan container images for vulnerabilities.
- Set CPU/memory limits.
- Monitor latency, error rates and prediction quality.
- Retrain and redeploy the model when model performance degrades.
