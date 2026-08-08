# Feature Engineering Impact Report

## 1. Objective

The goal of this experiment is to determine whether advanced feature engineering improves a baseline regression model on the UCI Wine Quality - Red dataset.

The techniques evaluated are:

1. Polynomial features and interaction terms.
2. Logarithmic transformations of selected skewed variables.
3. A combined feature-engineering approach.

## 2. Baseline

The baseline is a standardized Linear Regression model trained on the original physicochemical variables.

The baseline establishes the reference MAE, RMSE, and R² values.

## 3. Polynomial Features and Interactions

Degree-2 polynomial expansion creates squared terms and pairwise interactions.

### Expected benefit
The model can represent nonlinear effects and interactions that ordinary linear regression cannot express.

### Trade-offs
The number of features grows substantially. This can increase multicollinearity and overfitting risk, so Ridge regularization is used.

## 4. Mathematical Transformations

`log1p(x)` is applied to residual sugar, chlorides, free sulfur dioxide, and total sulfur dioxide.

These variables are non-negative and can be right-skewed. Logarithmic compression reduces the influence of extreme values and may produce a relationship that is easier for a linear model to learn.

## 5. Combined Engineering

The combined model uses both the logarithmic features and degree-2 polynomial/interaction expansion.

The notebook evaluates the combined model on the same held-out test set used by the baseline.

## 6. Performance Comparison

After running the notebook, the following file contains the actual measured values:

```text
docs/performance_comparison.csv
```

The notebook also creates:

```text
docs/metrics_summary.json
```

Do not manually invent metric values. Use the generated outputs when writing the final submission discussion.

## 7. Interpretability

Feature engineering improves model flexibility but can make individual coefficients harder to interpret.

- Original variables are directly understandable.
- Log features represent a transformed scale.
- Polynomial terms represent nonlinear effects.
- Interaction terms represent the joint effect of two variables.

Ridge regularization helps stabilize the engineered model but does not make every engineered coefficient directly interpretable in isolation.

## 8. Conclusion

The experiment demonstrates that feature engineering should be treated as an empirical process rather than an assumption that more features are always better.

Polynomial and interaction features increase the model's ability to capture nonlinear relationships, while logarithmic transformations can make skewed variables more suitable for linear modeling. The final choice should be based on held-out performance together with model complexity and interpretability.

The generated notebook contains the complete preprocessing, modeling, evaluation, visualization, and reproducibility workflow.
