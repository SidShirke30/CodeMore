# Model Performance Analysis

## Summary
The monitoring data shows gradual model degradation during the observed period.

## Findings

| Metric | Initial | Latest | Status |
|---|---:|---:|---|
| Accuracy | 0.95 | 0.87 | Degraded |
| Latency | 142 ms | 310 ms | Degraded |
| CPU | 51% | 72% | Within limit |
| Memory | 48% | 61% | Within limit |
| Error rate | 5% | 13% | Increased |

## Alert Conditions
An alert is triggered when accuracy falls below 0.90, latency exceeds 250 ms,
CPU or memory exceeds 85%, or accuracy drops more than 0.05 from baseline.

## Recommendations
1. Investigate changes in production data.
2. Review prediction latency and infrastructure load.
3. Check for data drift.
4. Retrain or roll back the model if degradation persists.
5. Continue monitoring after any model update.
