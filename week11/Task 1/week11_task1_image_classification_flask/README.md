# Week 11 Task 1 — Flask Image Classification API

## Objective
Deploy a trained image-classification model with Flask. The API accepts an uploaded image and returns the predicted class and confidence.

This project uses scikit-learn's built-in handwritten digits dataset (8×8 grayscale images), keeping the model lightweight and easy to run locally.

## Project Structure
```text
week11_task1_image_classification_flask/
├── app/
│   ├── app.py
│   ├── model.joblib
│   └── train_model.py
├── docs/
│   └── deployment.md
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

The serialized model is already included. To retrain it:
```bash
python app/train_model.py
```

## Run the Flask API
```bash
python app/app.py
```

The server starts at `http://127.0.0.1:5000`.

## Endpoints

### Health check
```text
GET /health
```

Example:
```bash
curl http://127.0.0.1:5000/health
```

### Image prediction
```text
POST /predict
Content-Type: multipart/form-data
Field: image
```

Example:
```bash
curl -X POST -F "image=@path/to/digit.png" http://127.0.0.1:5000/predict
```

Example response:
```json
{
  "predicted_class": 7,
  "confidence": 0.98
}
```

## Input
The uploaded image is converted to grayscale, resized to 8×8, normalized to the same scale used by the training data, and passed to the model.

## Notes
This is a production-like educational deployment. A real image-classification product would normally use a CNN/transfer-learning model, stronger validation, authentication, request limits, structured logging, and containerized deployment.
