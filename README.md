# Credit Default Prediction — MLOps Project

## Project Overview

This project develops a machine learning pipeline for predicting credit card payment defaults. It demonstrates an end-to-end MLOps workflow, including data validation, model training, experiment tracking, automated testing, model serving, and data drift monitoring.

## Dataset

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

- Original dataset: 30,000 records
- Features after preprocessing: 23
- Target: `default.payment.next.month`
- Target classes:
  - `0` — no default
  - `1` — default
- The `ID` column is removed during preprocessing.

## Current Implementation

- Data preprocessing and basic quality checks
- Data validation with Pandera
- Scikit-learn preprocessing and modeling pipeline
- Logistic Regression with hyperparameter tuning using GridSearchCV
- Experiment tracking and metric logging with MLflow
- Model serialization using Joblib
- Automated tests with pytest

## Initial Model Results

The initial Logistic Regression experiment achieved the following results on a held-out test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.6797 |
| Precision | 0.3672 |
| Recall | 0.6202 |
| F1-score | 0.4613 |
| ROC-AUC | 0.7081 |
| Cross-validation ROC-AUC | 0.7272 |

These results represent a baseline and will be compared with additional experiments.

## Project Structure

```text
src/
  data/
    prepare.py
    validation.py
  models/
    train.py
tests/
data/
  raw/
  processed/
models/
reports/
notebooks/
```

## Planned Improvements

- Compare at least five MLflow experiments
- Track datasets and model artifacts with DVC
- Add automated tests and GitHub Actions CI
- Build a FastAPI prediction service
- Containerize the service with Docker
- Implement data drift monitoring using the Population Stability Index (PSI)
- Complete project documentation and presentation

## Running the Project

Prepare the dataset:

```bash
python src/data/prepare.py
```

Validate the processed data:

```bash
python -m src.data.validation
```

Train the baseline model:

```bash
python -m src.models.train
```
