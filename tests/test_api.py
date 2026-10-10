from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 5000,
        "BILL_AMT2": 4500,
        "BILL_AMT3": 4000,
        "BILL_AMT4": 3500,
        "BILL_AMT5": 3000,
        "BILL_AMT6": 2500,
        "PAY_AMT1": 5000,
        "PAY_AMT2": 5000,
        "PAY_AMT3": 4000,
        "PAY_AMT4": 4000,
        "PAY_AMT5": 3000,
        "PAY_AMT6": 3000,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["default_prediction"] in [0, 1]
    assert 0 <= result["default_probability"] <= 1
