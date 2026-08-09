# Final Capstone Project Proposal
## Fair and Explainable Customer Churn Prediction Platform

**Week:** 9  
**Project type:** End-to-end machine learning capstone  
**Primary task:** Binary classification  
**Proposed domain:** Customer retention / business analytics

---

## 1. Executive Summary

The proposed capstone will build an end-to-end customer churn prediction platform. The system will learn from historical customer information and predict whether a customer is likely to discontinue a service.

The project will go beyond simply training a classifier. It will include data cleaning, feature engineering, baseline modeling, ensemble learning, systematic hyperparameter optimization, robust evaluation, model explainability, fairness analysis, and API deployment.

The final system will provide:
1. A trained and optimized classification model.
2. A reproducible preprocessing and prediction pipeline.
3. Performance and error analysis.
4. Explainability using SHAP or a comparable model-interpretation technique.
5. Fairness analysis across selected demographic/proxy groups where appropriate and ethically justified.
6. A Flask or FastAPI endpoint for real-time predictions.
7. Complete technical documentation.

---

## 2. Problem Statement

Customer churn can cause substantial revenue loss because replacing an existing customer is often more expensive than retaining one.

A business may have historical information such as:
- Customer tenure
- Contract type
- Monthly and total charges
- Service usage
- Support interactions
- Payment method
- Product/service subscriptions
- Customer satisfaction indicators

The challenge is to identify customers who are at elevated risk of churn early enough for the business to take appropriate retention action.

A useful solution must balance predictive performance with interpretability, reliability, fairness, and deployment practicality. A model that has high aggregate accuracy but performs poorly for an important customer segment may not be appropriate for production.

---

## 3. Proposed Solution

Build a machine learning pipeline that:

1. Loads and validates a tabular customer dataset.
2. Handles missing values and inconsistent data.
3. Encodes categorical variables and scales numerical variables when required.
4. Performs feature engineering to expose useful relationships.
5. Establishes a simple baseline classifier.
6. Compares stronger models such as:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - HistGradientBoosting or another suitable boosting model
7. Tunes selected models using GridSearchCV, RandomizedSearchCV, or Bayesian optimization.
8. Evaluates the final model on a held-out test set.
9. Generates confusion matrices, ROC curves, and Precision-Recall curves.
10. Explains model predictions using SHAP or an equivalent method.
11. Checks group-level performance and fairness metrics where protected/group attributes are available and appropriate to use.
12. Serializes the complete prediction pipeline.
13. Serves predictions through a Flask/FastAPI endpoint.
14. Documents limitations, ethical considerations, and deployment risks.

---

## 4. Dataset Description

### Proposed dataset

The initial dataset will be a public customer churn dataset, such as the IBM Telco Customer Churn dataset or another documented public tabular churn dataset.

The final dataset selection will be documented before implementation.

### Target variable

`Churn`

Binary target:
- `1` — customer churned
- `0` — customer remained

### Candidate feature categories

**Customer profile**
- Tenure
- Customer segment
- Demographic attributes where ethically appropriate

**Account information**
- Contract type
- Billing method
- Payment method
- Monthly charges
- Total charges

**Services**
- Internet/service plan
- Additional products
- Technical support
- Security or backup services

### Data preparation

The pipeline will:
- Detect missing values.
- Remove or correct invalid records.
- Convert numeric fields stored as text.
- Encode categorical variables.
- Separate features from target.
- Split data into training, validation/CV, and final test portions.
- Avoid data leakage by fitting preprocessing only on training folds.

---

## 5. Feature Engineering

Feature engineering will be guided by domain reasoning rather than arbitrary transformations.

Candidate engineered features include:

- Average monthly revenue = total charges / tenure.
- Tenure bands.
- Service-count features.
- Number of optional services.
- Contract-risk indicators.
- Interaction between tenure and contract type.
- Log transformation of highly skewed monetary variables when appropriate.

Polynomial or interaction features may be tested selectively when they provide measurable benefit without creating unnecessary dimensionality.

All transformations will be implemented inside reproducible scikit-learn pipelines where possible.

---

## 6. Modeling Strategy

### Baseline

A Logistic Regression model will provide an interpretable benchmark.

### Candidate models

The project will compare at least three approaches:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

An additional boosting model may be included if computational resources permit.

The comparison will consider predictive quality, interpretability, training time, inference cost, and complexity.

---

## 7. Hyperparameter Optimization

The best candidate models will be tuned using cross-validation.

Example search parameters:

### Random Forest
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

### Gradient Boosting
- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_split`
- `subsample`

### Logistic Regression
- `C`
- `solver`
- `class_weight`

RandomizedSearchCV will be considered when the search space becomes large. Bayesian optimization may be explored if it provides a meaningful computational advantage.

The final search configuration and best parameters will be documented.

---

## 8. Evaluation Metrics

Because churn datasets can be imbalanced, accuracy alone will not determine the winning model.

### Primary metrics

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

### Additional analysis

- Confusion matrix
- Classification report
- ROC curve
- Precision-Recall curve
- Calibration where appropriate
- Training/inference time

Recall will be particularly important because failing to identify a customer who is likely to churn may have a significant business cost.

The final threshold will be selected based on business objectives rather than automatically assuming 0.50 is optimal.

---

## 9. Explainability

SHAP will be used where technically appropriate to understand model behavior.

The analysis will include:

### Global explanation
- Most influential features.
- Direction and magnitude of feature contributions.
- Summary plots.

### Local explanation
- Why an individual customer received a high-risk prediction.
- Which features pushed the prediction toward churn or retention.

Explainability results will be translated into business-readable observations rather than presenting only technical plots.

---

## 10. Fairness and Ethical Considerations

Customer churn prediction can influence who receives retention offers, discounts, support, or other interventions.

Potential risks include:
- Historical bias in customer records.
- Unequal model performance between groups.
- Proxy variables indirectly representing protected characteristics.
- Unequal false-positive or false-negative rates.
- Using predictions to unfairly penalize customers.

Where appropriate and legally/ethically justified, the project will compare group-level:
- Selection/prediction rates
- Precision
- Recall
- False-positive rate
- True-positive rate

No protected attribute will be used for a real-world decision without appropriate authorization and ethical review.

The project will explicitly distinguish between fairness analysis for auditing and using sensitive attributes operationally.

---

## 11. Deployment Plan

The final trained preprocessing + model pipeline will be serialized.

A lightweight Flask or FastAPI service will expose:

### Health endpoint

`GET /health`

Example response:

```json
{
  "status": "ok"
}
```

### Prediction endpoint

`POST /predict`

Example request:

```json
{
  "tenure": 12,
  "monthly_charges": 79.5,
  "total_charges": 954.0,
  "contract": "Month-to-month"
}
```

The actual request schema will match the final selected dataset.

Example response:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.82
}
```

Input validation and graceful error handling will be included.

---

## 12. System Architecture

```text
Raw Dataset
     |
     v
Data Validation
     |
     v
Preprocessing
     |
     v
Feature Engineering
     |
     v
Train / Cross-Validation
     |
     +--> Baseline Model
     |
     +--> Ensemble Models
     |
     v
Hyperparameter Optimization
     |
     v
Final Test Evaluation
     |
     +--> Metrics
     +--> Error Analysis
     +--> Explainability
     +--> Fairness Audit
     |
     v
Serialized Pipeline
     |
     v
Flask / FastAPI
     |
     v
Prediction API
```

---

## 13. Success Criteria

The project will be considered successful if it:

1. Produces a reproducible end-to-end training pipeline.
2. Establishes a defensible baseline.
3. Demonstrates measurable improvement through model selection/tuning.
4. Reports multiple classification metrics.
5. Addresses class imbalance appropriately.
6. Includes explainability analysis.
7. Performs a documented fairness audit where applicable.
8. Provides a working local prediction API.
9. Includes clear setup and execution documentation.
10. Discusses limitations and responsible use.

The exact performance target will be established after dataset inspection so that the target is realistic and not artificially chosen.

---

## 14. Project Deliverables

The final capstone repository is expected to contain:

```text
capstone/
├── README.md
├── requirements.txt
├── data/
│   └── dataset_info.md
├── notebooks/
│   └── capstone_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── fairness.py
├── app/
│   └── main.py
├── models/
│   └── final_pipeline.joblib
├── reports/
│   ├── model_evaluation.md
│   ├── explainability_report.md
│   └── fairness_report.md
└── tests/
    └── test_api.py
```

---

## 15. Timeline

### Phase 1 — Data and baseline
- Dataset selection
- Data profiling
- Cleaning
- Baseline model

### Phase 2 — Feature engineering
- Feature creation
- Feature validation
- Leakage checks

### Phase 3 — Model comparison
- Logistic Regression
- Random Forest
- Gradient Boosting

### Phase 4 — Optimization
- Cross-validation
- Hyperparameter search
- Threshold optimization

### Phase 5 — Evaluation
- Test-set metrics
- Error analysis
- ROC/PR analysis

### Phase 6 — Explainability and fairness
- SHAP analysis
- Group-level evaluation
- Bias/fairness report

### Phase 7 — Deployment
- Serialize final pipeline
- Build API
- Test API
- Document deployment

### Phase 8 — Final documentation
- Final report
- README
- Results summary
- Limitations and future work

---

## 16. Risks and Limitations

- Public datasets may not represent a real company's customer population.
- Historical labels can encode existing business bias.
- A high-performing model can still be poorly calibrated.
- Fairness metrics may conflict with each other.
- Explainability methods are approximations and should not be treated as causal explanations.
- Model performance may degrade after deployment because customer behavior changes.
- The API demonstration is production-like but does not by itself establish production security, scalability, or compliance.

---

## 17. Concepts Demonstrated From the Internship

This capstone consolidates the major concepts covered during the internship:

| Internship concept | Capstone application |
|---|---|
| Data preprocessing | Missing values, encoding, scaling |
| Feature engineering | Interactions, ratios, transformations |
| Classical ML | Logistic Regression |
| Ensemble learning | Random Forest and Gradient Boosting |
| Hyperparameter tuning | Grid/Random/Bayesian search |
| Cross-validation | Robust model selection |
| Evaluation | Precision, Recall, F1, ROC-AUC, PR-AUC |
| Imbalanced learning | Threshold and metric analysis |
| Explainable AI | SHAP |
| AI ethics | Bias and fairness audit |
| Model serialization | Joblib pipeline |
| Model deployment | Flask/FastAPI prediction API |

---

## 18. Expected Final Outcome

The final product will be a reproducible, explainable, fairness-aware churn prediction system rather than a single trained model.

The project will demonstrate that a useful ML solution requires more than maximizing accuracy: data quality, leakage prevention, model selection, optimization, threshold selection, interpretability, fairness, responsible use, and deployment considerations must all be addressed.

The capstone will therefore serve as an integrated demonstration of the machine learning concepts developed throughout the internship.
