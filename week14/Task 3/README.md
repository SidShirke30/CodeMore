# Week 14 Task 3 — Model Retraining Strategy

## Objective
Develop a practical retraining strategy based on monitored production data. The project demonstrates retraining triggers, automated retraining, and evaluation of the new model before promotion.

## Retraining triggers
- Performance degradation: accuracy below the configured threshold.
- Data drift: mean feature shift above the configured threshold.
- Scheduled retraining: periodic retraining can be enabled separately.

## Workflow
1. Read monitored production metrics and new labeled data.
2. Check configured retraining triggers.
3. Train a candidate model when a trigger fires.
4. Evaluate the candidate against the current model.
5. Promote the candidate only when it meets the minimum improvement requirement.
6. Record the retraining decision and metrics.

## Run
```bash
pip install -r requirements.txt
python app/retrain.py
python dashboard/retraining_dashboard.py
pytest
```

See `docs/retraining_strategy.md` for the strategy and `reports/retraining_report.md` for the example evaluation.
