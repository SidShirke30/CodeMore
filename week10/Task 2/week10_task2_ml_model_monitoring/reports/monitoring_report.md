# Model Monitoring Report

## Current Metrics

| Metric | Value |
|---|---:|
| Prediction requests | 0 |
| Accuracy | N/A — no actual labels logged |
| Average inference latency | N/A |

## Prediction Distribution

```text
{}
```

## Drift Detection

The monitoring module includes a PSI-style function for comparing a reference prediction distribution with the current production distribution. A drift signal should trigger investigation rather than automatic retraining.
