from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "credit_default.csv"
TARGET_COLUMN = "default.payment.next.month"


def build_pipeline(model):
    """Create a preprocessing and modeling pipeline."""
    numeric_features = [
        column
        for column in pd.read_csv(DATA_PATH, nrows=0).columns
        if column != TARGET_COLUMN
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", model),
        ]
    )


def run_experiments():
    """Train and compare four additional model configurations."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("credit-default-prediction")

    df = pd.read_csv(DATA_PATH)

    from src.data.validation import validate_data

    df = validate_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    experiments = [
        (
            "Logistic Regression C=1 balanced",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        (
            "Logistic Regression C=10 unbalanced",
            LogisticRegression(
                C=10.0,
                class_weight=None,
                max_iter=2000,
                random_state=42,
            ),
        ),
        (
            "Random Forest baseline",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Extra Trees baseline",
            ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    for experiment_name, model in experiments:
        pipeline = build_pipeline(model)

        with mlflow.start_run(run_name=experiment_name):
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                pipeline,
                name="credit_default_model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            print(f"\nExperiment: {experiment_name}")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    run_experiments()
