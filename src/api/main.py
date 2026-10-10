from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "credit_default_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Credit Default Prediction API",
    description="API for predicting credit card default risk.",
    version="1.0.0",
)


class CreditApplication(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=1, le=6)
    MARRIAGE: int = Field(ge=1, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/")
def root():
    return {"message": "Credit Default Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(application: CreditApplication):
    features = pd.DataFrame([application.model_dump()])
    prediction = int(model.predict(features)[0])

    result = {"default_prediction": prediction}

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]
        result["default_probability"] = float(probability)

    return result
