from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/credit_default_model.pkl")

app = FastAPI(title="Credit Default Prediction API")

model = load(MODEL_PATH)


class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
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
    return {"status": "ok", "message": "Credit Default Prediction API is alive!"}


@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.model_dump()])

    df["AGE_BIN"] = pd.cut(
        df["AGE"], bins=[0, 25, 35, 45, 55, 100], labels=False, include_lowest=True
    )
    df["PAY_AMT1_TO_LIMIT"] = df["PAY_AMT1"] / (df["LIMIT_BAL"] + 1e-6)
    df["TOTAL_DELAYED"] = (df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] > 0).sum(
        axis=1
    )

    y_pred = model.predict(df)[0]
    y_proba = model.predict_proba(df)[0][1]
    return {
        "default_prediction": int(y_pred),
        "default_probability": float(y_proba),
    }
