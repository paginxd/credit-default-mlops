import pandas as pd
import pandera.pandas as pa

TARGET_COLUMN = "default.payment.next.month"

EXPECTED_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    TARGET_COLUMN,
]


schema = pa.DataFrameSchema(
    {
        "LIMIT_BAL": pa.Column(float, checks=pa.Check.ge(0)),
        "SEX": pa.Column(int),
        "EDUCATION": pa.Column(int),
        "MARRIAGE": pa.Column(int),
        "AGE": pa.Column(int, checks=pa.Check.gt(0)),
        "PAY_0": pa.Column(int),
        "PAY_2": pa.Column(int),
        "PAY_3": pa.Column(int),
        "PAY_4": pa.Column(int),
        "PAY_5": pa.Column(int),
        "PAY_6": pa.Column(int),
        "BILL_AMT1": pa.Column(float),
        "BILL_AMT2": pa.Column(float),
        "BILL_AMT3": pa.Column(float),
        "BILL_AMT4": pa.Column(float),
        "BILL_AMT5": pa.Column(float),
        "BILL_AMT6": pa.Column(float),
        "PAY_AMT1": pa.Column(float, checks=pa.Check.ge(0)),
        "PAY_AMT2": pa.Column(float, checks=pa.Check.ge(0)),
        "PAY_AMT3": pa.Column(float, checks=pa.Check.ge(0)),
        "PAY_AMT4": pa.Column(float, checks=pa.Check.ge(0)),
        "PAY_AMT5": pa.Column(float, checks=pa.Check.ge(0)),
        "PAY_AMT6": pa.Column(float, checks=pa.Check.ge(0)),
        TARGET_COLUMN: pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
        ),
    },
    strict=True,
    coerce=True,
    checks=[
        pa.Check(
            lambda df: list(df.columns) == EXPECTED_COLUMNS,
            error="Dataset columns do not match the expected order",
        )
    ],
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the prepared credit default dataset."""
    return schema.validate(df)


if __name__ == "__main__":
    from pathlib import Path

    data_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "credit_default.csv"
    )

    data = pd.read_csv(data_path)
    validated_data = validate_data(data)

    print("Data validation passed!")
    print(f"Rows: {len(validated_data)}")
    print(f"Columns: {len(validated_data.columns)}")
