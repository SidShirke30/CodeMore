# Explainable AI Report — Week 13 Task 3

## 1. Objective

The objective is to make an individual machine learning prediction more interpretable using an Explainable AI technique.

The project uses **LIME (Local Interpretable Model-agnostic Explanations)** to explain predictions from a Random Forest classifier.

## 2. Model

A Random Forest classifier is trained on the Iris dataset. The four input features are:

- sepal length
- sepal width
- petal length
- petal width

The model predicts one of three Iris classes.

## 3. Why LIME?

LIME is model-agnostic, meaning it can explain predictions from many different model types without requiring access to the model's internal decision process.

For an individual input, LIME:

1. Generates perturbed samples around the selected instance.
2. Gets the model's predictions for those samples.
3. Learns a simple interpretable local model.
4. Uses feature weights to show which features contributed to the local prediction.

## 4. Specific Prediction

The implementation explains the first sample in `data/sample_data.csv`.

The script prints:

- the input feature values
- the predicted class
- class probabilities
- positive and negative local feature contributions

An interactive HTML explanation is generated at:

`explanations/prediction_explanation.html`

## 5. Interpretation

A positive LIME weight indicates that the corresponding local feature condition supports the explained class, while a negative weight indicates that it pushes the local explanation away from that class.

The explanation is **local**, so it describes the reasoning around the selected instance rather than providing a complete global explanation of the Random Forest.

## 6. Benefits

- Improves transparency of individual predictions.
- Helps developers inspect unexpected model behavior.
- Makes model behavior easier to communicate to non-technical stakeholders.
- Can support debugging and trust-building.

## 7. Limitations

LIME explanations depend on the local perturbation strategy and configuration. They should therefore be treated as an approximation of the model's local behavior rather than a guaranteed causal explanation.

For production systems, explanations should be validated across many representative instances and combined with model-level evaluation.

## 8. Conclusion

The project demonstrates a practical XAI workflow by applying LIME to a previously trained classification model. The generated local explanation provides insight into which features influenced a specific prediction and helps improve the transparency of the machine learning system.
