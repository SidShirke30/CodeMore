# Monitoring and Automated Retraining Report

## Monitoring approach
The system tracks classification accuracy on newly labeled data and calculates a simple standardized feature-mean drift score against the reference Iris dataset.

## Trigger rules
Retraining is requested when either:
- Accuracy falls below **0.90**, or
- Drift score exceeds **0.20**.

These thresholds are configurable in `app/monitoring.py`.

## Retraining strategy
The retraining script:
1. Loads the training dataset.
2. Creates a stratified train/test split.
3. Trains a Random Forest model.
4. Validates the new model.
5. Refuses to save the replacement model if validation accuracy is below 0.90.
6. Saves the validated model as `app/model.joblib`.

## Production recommendations
For a production system, add:
- Model and dataset versioning.
- Scheduled monitoring jobs.
- Alerting when thresholds are crossed.
- Shadow validation of candidate models.
- Approval gates before production replacement.
- Automatic rollback to the previous model.
- Feature-level drift metrics such as PSI or statistical tests.
- Separate training, staging and production environments.

## Conclusion
Automated monitoring and retraining can reduce model decay by identifying performance degradation early. Retraining should not automatically replace a production model without validation and rollback safeguards.
