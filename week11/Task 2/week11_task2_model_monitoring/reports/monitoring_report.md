# Model Monitoring Report

## 1. Objective
The monitoring system tracks accuracy, precision, and recall of the deployed image-classification model across multiple monitoring periods.

## 2. Observed Performance

| Period | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Week 1 | 0.920 | 0.918 | 0.925 |
| Week 2 | 0.900 | 0.895 | 0.910 |
| Week 3 | 0.870 | 0.865 | 0.880 |
| Week 4 | 0.780 | 0.775 | 0.790 |
| Week 5 | 0.740 | 0.735 | 0.750 |

## 3. Findings
Performance declines consistently over time. Accuracy falls from 0.92 to 0.74, while precision and recall show a similar downward trend.

This pattern is consistent with possible model decay. In a real production environment, the next step would be to investigate whether the input image distribution has changed, whether new classes or image conditions have appeared, or whether the labelled feedback data has changed.

## 4. Recommendations
1. Monitor input-feature/image distribution for drift.
2. Collect representative labelled production samples.
3. Retrain or fine-tune the model when degradation persists.
4. Establish alert thresholds for accuracy, precision, and recall.
5. Keep model and dataset versions with each monitoring record.
6. Review false positives and false negatives regularly.

## 5. Conclusion
The monitoring pipeline provides a repeatable method for identifying declining model performance. The simulated results trigger a decay warning because the latest accuracy is substantially below the baseline.
