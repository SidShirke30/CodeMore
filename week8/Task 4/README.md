# Week 8 Task 4 - Ensemble Learning

Compare a single Decision Tree with Random Forest and Gradient Boosting on the Scikit-Learn Breast Cancer Wisconsin Diagnostic dataset.

## Structure
```text
Task 4/
├── README.md
├── requirements.txt
├── data/raw_dataset_info.txt
├── notebooks/ensemble_exploration.ipynb
└── reports/performance_comparison.md
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Open `notebooks/ensemble_exploration.ipynb` and run all cells.

The notebook performs preprocessing checks, a stratified train/test split, trains a Decision Tree, Random Forest and Gradient Boosting model, and compares Accuracy, Precision, Recall, F1, ROC-AUC, training time and train/test performance gaps.
