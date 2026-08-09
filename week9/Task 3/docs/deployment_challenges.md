# Deployment Challenges and Mitigation Strategies

## 1. Dependency Management
A model can fail to load if the production environment uses incompatible versions of Python libraries.

**Mitigation:** pin dependency versions in `requirements.txt`, test from a clean virtual environment, and use a lockfile or container image for larger deployments.

## 2. Environment Consistency
Differences between development and production operating systems, Python versions, or CPU libraries can cause unexpected behavior.

**Mitigation:** reproduce the same environment using virtual environments, containers, or a managed deployment platform.

## 3. Model Loading Latency
Loading a serialized model for every request increases latency.

**Mitigation:** load the model once during application startup, as implemented in `app.py`.

## 4. Input Validation
Malformed or unexpected JSON can cause inference errors.

**Mitigation:** validate the request type, feature count, and numeric values before calling the model.

## 5. Development Server vs Production
Flask's built-in development server is not designed for production traffic.

**Mitigation:** use a production WSGI server such as Gunicorn or Waitress and place it behind a reverse proxy when appropriate.

## 6. Monitoring
Production services need visibility into errors, latency, traffic, and model behavior.

**Mitigation:** add structured logging, health checks, metrics, error tracking, and model-performance monitoring.

## 7. Security
An unrestricted prediction API can be abused or overwhelmed.

**Mitigation:** add authentication, authorization, rate limiting, request-size limits, HTTPS, and appropriate network controls.

## Local Test Commands

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Prediction:

```powershell
$body = '{"features":[5.1,3.5,1.4,0.2]}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body $body
```

Invalid request:

```powershell
$body = '{"features":[1,2]}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body $body
```

The final command should return a clear HTTP 400 validation error.
