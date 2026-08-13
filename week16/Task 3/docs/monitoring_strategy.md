# Monitoring Strategy

## KPIs
### Accuracy
Track prediction correctness on labeled production data. A sustained decline can indicate model drift or changes in the relationship between features and targets.

### Latency
Track inference response time. Rising latency can indicate increased traffic, inefficient inference, or insufficient compute capacity.

### Resource Utilization
Track CPU and memory usage to identify infrastructure pressure and capacity problems.

## Drift Detection
Use changes in accuracy and, in a production system, compare feature distributions between training and production data. Significant changes should trigger investigation.

## Alerting
The example implementation flags:
- accuracy below 85%
- latency above 250 ms
- CPU or memory above 80%
- accuracy decline greater than 5 percentage points from the baseline sample

These thresholds should be tuned to the model's business requirements.
