# API Usage

## Start the server

From the project root:

```powershell
python app/train_model.py
python app/app.py
```

## Health check

```powershell
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "model_loaded": true,
  "status": "ok"
}
```

## Prediction endpoint

### Endpoint
`POST /predict`

### Header
```text
Content-Type: application/json
```

### Body
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### PowerShell example

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"features":[5.1,3.5,1.4,0.2]}'
```

### Example response

```json
{
  "prediction": 0,
  "probabilities": [0.98, 0.02, 0.0]
}
```

## Error handling

The API returns appropriate HTTP errors for:
- Missing JSON content type
- Invalid JSON body
- Missing `features`
- Incorrect feature count
- Non-numeric feature values
- Prediction failures

## Dataset
The demonstration model is trained on the scikit-learn Iris dataset. The four input features are sepal length, sepal width, petal length, and petal width.
