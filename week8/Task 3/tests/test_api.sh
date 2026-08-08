#!/usr/bin/env bash
# Local API smoke test. Start the API first:
# uvicorn app.main:app --reload
#
# The feature vector below is the first observation from the sklearn dataset.
# The API expects exactly 30 numeric values.

echo "1) Health check"
curl -s http://127.0.0.1:8000/health
echo
echo

echo "2) Prediction request"
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
      1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
      25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
    ]
  }'
echo
