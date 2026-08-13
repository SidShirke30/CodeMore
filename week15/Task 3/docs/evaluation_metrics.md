# Model Improvement Evaluation Metrics

The retrained candidate must be compared with the current production model.

## Classification metrics

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of positive predictions that are correct.
- **Recall:** proportion of actual positives detected.
- **F1 score:** harmonic mean of precision and recall.

## Production metrics

- Average latency
- P95 latency
- Error rate
- Resource utilization

## Acceptance example

The candidate can proceed when:

- Accuracy is not lower than the current model.
- F1 score improves by at least 1%.
- Error rate does not increase.
- Latency remains within the production limit.
- A/B testing shows no significant regression.

The exact thresholds should be adapted to the business requirements of the
deployed model.
