import argparse
import sys
import pandas as pd
import great_expectations as ge


def create_suite(df: pd.DataFrame):
    ge_df = ge.from_pandas(df)

    ge_df.expect_column_to_exist("LIMIT_BAL")
    ge_df.expect_column_values_to_not_be_null("LIMIT_BAL")
    ge_df.expect_column_values_to_be_between("AGE", min_value=18, max_value=100)
    ge_df.expect_column_values_to_be_in_set(
        "default.payment.next.month", value_set=[0, 1]
    )
    ge_df.expect_column_values_to_be_between("EDUCATION", min_value=0, max_value=6)

    return ge_df


def validate(path: str) -> bool:
    df = pd.read_csv(path)
    ge_df = create_suite(df)
    res = ge_df.validate()
    return res["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="CSV to validate")
    args = parser.parse_args()

    ok = validate(args.path)
    if not ok:
        print(f"[validation] Validation FAILED for {args.path}")
        sys.exit(1)
    else:
        print(f"[validation] Validation PASSED for {args.path}")
