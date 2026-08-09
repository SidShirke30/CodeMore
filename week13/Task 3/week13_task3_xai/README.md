# Week 13 Task 3 — Explainable AI (XAI)

## Objective
Apply an Explainable AI technique to a machine learning classification model and explain why the model made specific predictions.

This project uses **LIME (Local Interpretable Model-agnostic Explanations)** with a Random Forest classifier trained on the Iris dataset.

## Deliverables
- Trained machine learning model
- LIME explanation pipeline
- Explanation for a specific prediction
- Sample dataset
- XAI report describing model decisions
- Reproducible requirements and execution instructions

## Project Structure
```text
Task 3/
├── app/
│   ├── explainability.py
│   └── train_model.py
├── data/
│   └── sample_data.csv
├── explanations/
│   └── prediction_explanation.html
├── reports/
│   └── xai_report.md
├── README.md
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Train the model
```bash
python app/train_model.py
```

## Generate a LIME explanation
```bash
python app/explainability.py
```

The generated explanation is saved as:
`explanations/prediction_explanation.html`

## XAI Interpretation
LIME explains an individual prediction by creating perturbed versions of the input, observing how the model responds, and fitting a simple interpretable local model. The resulting feature contributions show which input features supported or opposed the predicted class.

## Notes
The explanation is local to the selected prediction. It should not be interpreted as a complete description of the model's global behavior.
