# Flask ML API Usage

## Base URL
`http://127.0.0.1:5000`

## Health
```text
GET /health
```

## Prediction
```text
POST /predict
Content-Type: application/json
```

Body:
```json
{"features": [5.1, 3.5, 1.4, 0.2]}
```

Feature order:
1. sepal length
2. sepal width
3. petal length
4. petal width

Example response:
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [1.0, 0.0, 0.0]
}
```

## PowerShell
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"features":[5.1,3.5,1.4,0.2]}'
```
