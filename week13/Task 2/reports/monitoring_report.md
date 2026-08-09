# Monitoring Report

## Objective

The monitoring system tracks production model accuracy, inference latency, and feature distribution drift. Threshold-based alerts identify potential degradation before it becomes a larger production issue.

## Metrics

### Accuracy
Accuracy is monitored over time. A drop below the configured minimum of 0.85 generates an alert.

### Latency
Average inference latency is tracked in milliseconds. Values above 250 ms generate a performance alert.

### Data Drift
Population Stability Index (PSI) is used to compare production feature distributions with a baseline distribution.

Interpretation used in this project:
- PSI < 0.10: low/no meaningful drift
- 0.10–0.20: moderate drift
- > 0.20: significant drift requiring investigation

## Observations

The supplied production history shows declining accuracy and increasing latency toward the end of the monitoring period. The latest observations cross the configured accuracy and latency thresholds.

Feature distributions are also compared against the baseline. A high PSI value triggers a data-drift alert.

## Recommended Actions

1. Investigate the data pipeline when drift is detected.
2. Validate whether the production population has changed.
3. Review model errors by segment.
4. Retrain the model if degradation persists and sufficient labeled data is available.
5. Re-evaluate the replacement model before deployment.
6. Keep monitoring thresholds configurable rather than hard-coded.

## Production Considerations

A production implementation could send alerts to email, Slack, PagerDuty, or a cloud monitoring service. Metrics should be stored in a time-series system and dashboards should be connected to live telemetry.
