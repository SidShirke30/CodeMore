# Model Optimization Report

## Results

| Metric | Original | Optimized |
|---|---:|---:|
| Accuracy | 0.9233 | 0.8867 |
| Total latency (ms) | 33.712 | 24.296 |
| Latency / prediction (ms) | 0.05619 | 0.04049 |
| Model size (KB) | 7214.24 | 329.85 |

## Interpretation

- Accuracy change: **-0.0367**
- Latency change: **-27.93%**
- Model-size change: **-95.43%**

The optimized model uses fewer trees and a maximum tree depth of 8, reducing model complexity. Joblib compression further reduces the serialized artifact size.

## Deployment Recommendation

Accept the optimized model only if its accuracy remains within the required business threshold. Validate it on production-like data before replacing the deployed model.
