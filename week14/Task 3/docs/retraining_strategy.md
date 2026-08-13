# Model Retraining Strategy

## 1. Performance trigger
Start a retraining evaluation when rolling accuracy drops below 80%. Precision and recall should also be monitored to detect class-specific degradation.

## 2. Data-drift trigger
Monitor changes in production feature distributions compared with the training baseline. A drift score above the configured threshold starts a candidate retraining run.

## 3. Scheduled trigger
Run a periodic review every 30 days even if no alert has fired. This provides a safety net for gradual changes.

## 4. Retraining methods
- Incremental retraining when the model and data pipeline support it.
- Full retraining when the training dataset can be rebuilt reliably.
- Hyperparameter tuning when the data distribution is stable but model performance has declined.

## 5. Promotion policy
Never replace the production model solely because a trigger fired. Train a candidate, evaluate it on validation data, compare it with the current model, and promote only when the configured improvement threshold is met.

## 6. Rollback
Keep the previous production model and metadata so that a failed deployment can be reversed quickly.

## 7. Audit trail
Store trigger reason, metrics, dataset version, model version, timestamp, and promotion decision for each retraining event.
