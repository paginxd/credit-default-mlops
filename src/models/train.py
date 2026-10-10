import mlflow
import mlflow.sklearn

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "credit_default.csv"
MODEL_PATH = ROOT_DIR / "models" / "credit_default_model.joblib"
TARGET_COLUMN = "default.payment.next.month"


def train_model() -> None:

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("credit-default-prediction")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Validate data before training.
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

    numeric_features = X.columns.tolist()

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

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run():
        print("Training model and tuning hyperparameters...")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_roc_auc", search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model,
            name="credit_default_model",
            skops_trusted_types=["numpy.dtype"],
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nCross-validation ROC-AUC:")
        print(round(search.best_score_, 4))

        print("\nTest metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print(f"Model saved to: {MODEL_PATH}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_model()
