# A/B Testing Strategy

## Experiment design
Traffic is split between model versions A (control) and B (treatment). Requests are assigned consistently and outcomes are logged with model version, correctness, and latency.

## Metrics
1. **Accuracy** — primary quality metric.
2. **Latency** — average response time in milliseconds.
3. **Error rate** — proportion of incorrect predictions.
4. **Sample size** — used to judge whether the comparison is sufficiently powered.

## Decision rule
Use a two-sided two-proportion z-test for accuracy with alpha = 0.05. Prefer B when its accuracy improvement is statistically significant and it does not create an unacceptable latency regression. Otherwise retain A.

## Rollout
Start with a small treatment percentage, evaluate results, then increase B traffic gradually if the quality and latency gates remain healthy. Keep rollback available throughout the experiment.
