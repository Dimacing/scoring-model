import pandas as pd
from src.models.pipeline import create_pipeline


def test_pipeline_fits():
    df = pd.DataFrame(
        {
            "LIMIT_BAL": [20000, 120000, 90000],
            "AGE": [24, 26, 34],
            "BILL_AMT1": [3913, 2682, 29239],
            "BILL_AMT2": [3102, 1725, 14027],
            "BILL_AMT3": [689, 2682, 13559],
            "BILL_AMT4": [0, 3272, 14331],
            "BILL_AMT5": [0, 3455, 14948],
            "BILL_AMT6": [0, 3261, 15549],
            "PAY_AMT1": [0, 0, 1518],
            "PAY_AMT2": [689, 1000, 1500],
            "PAY_AMT3": [0, 1000, 1000],
            "PAY_AMT4": [0, 1000, 1000],
            "PAY_AMT5": [0, 0, 1000],
            "PAY_AMT6": [0, 2000, 5000],
            "SEX": [2, 2, 2],
            "EDUCATION": [2, 2, 2],
            "MARRIAGE": [1, 2, 2],
            "PAY_0": [2, -1, 0],
            "PAY_2": [2, 2, 0],
            "PAY_3": [-1, 0, 0],
            "PAY_4": [-1, 0, 0],
            "PAY_5": [-2, 0, 0],
            "PAY_6": [-2, 2, 0],
            "AGE_BIN": [1, 1, 2],
            "PAY_AMT1_TO_LIMIT": [0, 0, 1518 / 90000],
            "TOTAL_DELAYED": [2, 2, 0],
            "default.payment.next.month": [1, 1, 0],
        }
    )

    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]

    pipe = create_pipeline()
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    assert len(proba) == 3
