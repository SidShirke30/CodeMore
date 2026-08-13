# Cloud Deployment Guide

## 1. Build and test locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python app/train_model.py
```

Run the API:

```bash
python app/app.py
```

Check:

```text
GET http://localhost:5000/health
```

Send a prediction:

```bash
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[5.1,3.5,1.4,0.2]}"
```

## 2. Container deployment

Build the image:

```bash
docker build -t week16-ml-api .
```

Run it:

```bash
docker run -p 5000:5000 week16-ml-api
```

The same container can be pushed to a cloud container registry and deployed on a managed container platform.

## 3. Generic managed-cloud flow

1. Create a project/application in the chosen cloud platform.
2. Create a container registry.
3. Build and push the Docker image.
4. Create a managed container/web service from that image.
5. Configure the service to expose the application port.
6. Set health checks to `/health`.
7. Deploy and obtain the public service URL.
8. Test `/predict` using the public URL.
9. Configure authentication, HTTPS, logging, autoscaling, and resource limits for production.

## 4. Production considerations

- Keep the model artifact versioned.
- Do not expose sensitive training data.
- Add authentication and rate limiting.
- Enable HTTPS.
- Monitor latency, errors, request volume, and resource utilization.
- Configure minimum/maximum instances according to traffic.
- Use CI/CD to test before deploying a new image.
