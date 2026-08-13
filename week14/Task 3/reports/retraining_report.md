# Model Retraining Evaluation Report

## Strategy
The model is evaluated using production labels when they become available. Retraining is considered when accuracy falls below `0.80` or when the configured drift score exceeds `0.50`.

## Candidate validation
A newly trained candidate is evaluated on a held-out portion of the latest labeled data. The candidate is promoted only if its accuracy improves on the current model by at least `0.01`.

## Example result
The sample history shows a retraining event where current accuracy was `0.76` and the candidate reached `0.84`. Because the improvement exceeded the configured promotion threshold, the candidate was promoted.

## Recommended production policy
- Use rolling windows instead of a single batch.
- Monitor accuracy, precision, recall, latency, and drift.
- Retrain on sustained degradation rather than one noisy observation.
- Validate the candidate before deployment.
- Keep the previous model for rollback.
- Record model, dataset, and configuration versions for every retraining event.
