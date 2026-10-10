from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "credit_default.csv"

TARGET_COLUMN = "default.payment.next.month"
ID_COLUMN = "ID"


def prepare_data() -> None:
    """Validate the source dataset and save data without the ID column."""
    df = pd.read_csv(RAW_DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")

    if ID_COLUMN not in df.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' is missing")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1")

    df = df.drop(columns=[ID_COLUMN])

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Prepared dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())
    print(f"Missing values: {df.isna().sum().sum()}")


if __name__ == "__main__":
    prepare_data()
