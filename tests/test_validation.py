import pandas as pd
import pytest

from src.data.validation import validate_data

DATA_PATH = "data/processed/credit_default.csv"


def test_prepared_data_passes_validation():
    df = pd.read_csv(DATA_PATH)

    validated_df = validate_data(df)

    assert validated_df.shape == (30000, 24)


def test_validation_rejects_invalid_target():
    df = pd.read_csv(DATA_PATH)
    df.loc[0, "default.payment.next.month"] = 5

    with pytest.raises(Exception):
        validate_data(df)


def test_validation_rejects_missing_column():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["AGE"])

    with pytest.raises(Exception):
        validate_data(df)
