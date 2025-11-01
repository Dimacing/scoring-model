import pandas as pd
from src.data.validation import validate


def test_validation_on_sample(tmp_path):
    sample = pd.DataFrame(
        {
            "LIMIT_BAL": [20000, 120000],
            "AGE": [24, 30],
            "EDUCATION": [2, 2],
            "MARRIAGE": [1, 2],
            "default.payment.next.month": [1, 0],
        }
    )
    fpath = tmp_path / "sample.csv"
    sample.to_csv(fpath, index=False)

    assert validate(str(fpath)) is True
