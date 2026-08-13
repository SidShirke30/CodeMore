# New Data Collection Pipeline

## Pipeline stages

1. **Capture** — collect inputs, predictions, timestamps, model version, and
   later-arriving ground-truth labels.
2. **Queue** — temporarily buffer incoming records.
3. **Validate** — check schema, missing values, ranges, duplicates, and labels.
4. **Version** — assign a dataset version and record its creation metadata.
5. **Store** — keep approved records in the training-data store.
6. **Trigger** — notify the retraining system when a configured condition is met.
7. **Train** — create a candidate model from the approved dataset.

## Example record

```json
{
  "timestamp": "2026-08-13T18:00:00",
  "model_version": "v1.2",
  "features": [0.31, 0.72, 0.15],
  "prediction": 1,
  "actual_label": 1
}
```

## Data quality rules
- Required columns must be present.
- Numeric features must have valid ranges.
- Duplicate records should be removed.
- Missing labels should not be used for supervised retraining.
- Dataset versions must be immutable after approval.
