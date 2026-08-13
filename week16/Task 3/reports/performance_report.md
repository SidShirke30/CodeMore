# Model Performance Analysis

## Summary
The collected metrics show a gradual decline in model quality alongside increases in latency and resource utilization.

## Findings
- Accuracy declined from **94% to 84%**, crossing the 85% monitoring threshold.
- Latency increased from **120 ms to 275 ms**, indicating slower inference.
- CPU utilization reached **83%** and memory utilization reached **82%**, indicating increased resource pressure.
- The accuracy decline is greater than 5 percentage points, so the monitoring system flags potential model drift.

## Recommended Actions
1. Investigate incoming data for distribution changes.
2. Compare production feature distributions with the training baseline.
3. Validate the model on recent labeled production data.
4. Retrain if drift or sustained accuracy degradation is confirmed.
5. Review infrastructure capacity and optimize inference latency.
6. Continue monitoring after remediation to verify that performance recovers.

## Monitoring Thresholds
- Accuracy: alert below 85%
- Latency: alert above 250 ms
- CPU: alert above 80%
- Memory: alert above 80%
