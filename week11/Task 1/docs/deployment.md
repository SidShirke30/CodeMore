# Deployment Documentation

## Architecture
1. Flask starts and loads `app/model.joblib` once.
2. A client uploads an image to `POST /predict`.
3. The server validates the upload.
4. PIL converts the image to grayscale and resizes it to 8×8.
5. The serialized classifier returns the predicted digit and confidence.
6. Flask returns JSON.

## Production Considerations
- Use Gunicorn or another production WSGI server instead of Flask's development server.
- Pin dependencies and build the application in a reproducible environment.
- Add authentication and HTTPS for protected services.
- Add structured request/error logging and latency monitoring.
- Apply file-size limits and content validation to uploaded images.
- Keep model artifacts versioned and support rollback.
- Add automated tests and CI/CD before deployment.
