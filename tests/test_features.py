import pandas as pd
from src.data.build_features import feature_engineering


def test_feature_engineering_creates_columns():
    df = pd.DataFrame(
        {
            "LIMIT_BAL": [20000],
            "AGE": [30],
            "PAY_AMT1": [1000],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [0],
            "PAY_4": [0],
            "PAY_5": [0],
            "PAY_6": [0],
        }
    )
    out = feature_engineering(df)
    assert "AGE_BIN" in out.columns
    assert "PAY_AMT1_TO_LIMIT" in out.columns
    assert "TOTAL_DELAYED" in out.columns
