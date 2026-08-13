# Retraining Strategy

## 1. Trigger Detection

Retraining should not happen after every new prediction. It should be
triggered by meaningful evidence of model degradation or by a controlled
schedule.

Recommended triggers:

| Trigger | Example threshold |
|---|---|
| Accuracy degradation | Accuracy < 0.90 |
| Error-rate increase | Error rate > 0.10 |
| Data drift | Drift score > 0.20 |
| New labeled data | At least 100 new samples |
| Scheduled retraining | Weekly |

A trigger creates a retraining job rather than immediately replacing the
production model.

## 2. Data Collection

Collect prediction inputs, predictions, timestamps, model version, and labels
when they become available. Remove invalid records and protect sensitive data.

## 3. Data Validation and Versioning

Validate schema, missing values, ranges, duplicates, and label quality.
Store each training dataset with a version identifier so that a model can be
reproduced.

## 4. Retraining

Combine validated historical data with approved new data. Train a candidate
model using the same reproducible preprocessing and feature pipeline.

## 5. Evaluation

Evaluate both the current model and candidate model on a fixed holdout set.
Track accuracy, precision, recall, F1 score, and latency.

## 6. A/B Testing

If offline evaluation is successful, expose a controlled percentage of
traffic to the candidate. Compare production metrics against the current
model.

## 7. Deployment and Rollback

Deploy only when acceptance criteria are met. Keep the previous model
available so the system can quickly roll back if production performance
degrades.

## 8. Online Learning

For high-volume use cases, incremental/online learning can update a model
more frequently. It should still use validation gates and monitoring to avoid
propagating noisy or incorrect data.
