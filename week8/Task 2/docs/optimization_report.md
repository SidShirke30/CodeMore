# Week 8 Task 2 - Optimization Report

## 1. Objective

This task compares two advanced hyperparameter optimization strategies:

1. Random Search
2. Bayesian Optimization using Optuna's Tree-structured Parzen Estimator (TPE)

The goal is to maximize Gradient Boosting classification performance while keeping search cost reasonable.

## 2. Model and Dataset

The experiment uses the Breast Cancer Wisconsin dataset available through Scikit-Learn and a Gradient Boosting classifier.

The same train/test split, cross-validation strategy, random seed, scoring metric, and 30-trial/iteration budget are used for both optimization approaches.

Primary optimization metric: **5-fold cross-validated ROC-AUC**.

Secondary test metrics:
- Accuracy
- F1-score
- ROC-AUC
- Execution time

## 3. Baseline

A baseline Gradient Boosting configuration should be evaluated before optimization.

The baseline provides a reference point for determining whether either search strategy provides a meaningful improvement.

## 4. Random Search

Random Search samples combinations from predefined parameter distributions.

Advantages:
- simple to implement,
- easy to parallelize,
- explores a broad search space,
- does not require a surrogate model.

Disadvantage:
- previous trials do not guide later trials.

The implementation performs 30 sampled configurations using 5-fold cross-validation.

## 5. Bayesian Optimization

Optuna's TPE sampler uses information from previous trials to select promising future configurations.

Advantages:
- learns from earlier evaluations,
- can focus evaluations on promising regions,
- often reaches strong configurations with fewer wasted trials.

The implementation performs 30 trials with the same cross-validation and objective metric as Random Search.

## 6. Convergence Comparison

The optimization histories should be compared using:
- best validation ROC-AUC versus trial/iteration,
- final validation score,
- test ROC-AUC,
- execution time.

A convergence plot can be generated from the saved trial/search histories.

## 7. Expected Interpretation

Random Search is a strong baseline for large, mixed hyperparameter spaces because it is inexpensive and independent across trials.

Bayesian Optimization can be more efficient when each model evaluation is expensive because the optimizer uses information from previous trials to guide subsequent sampling.

The final winner should be selected using both predictive performance and search efficiency rather than validation score alone.

## 8. Reproducibility

Both pipelines use `random_state=42` where applicable and a 5-fold cross-validation objective.

Run:

```bash
python src/random_search.py
python src/bayesian_opt.py
```

The scripts print the optimal parameters, validation score, test metrics, and execution time.

For a complete comparison, record the outputs from both scripts and place the resulting values in the final submission report.
