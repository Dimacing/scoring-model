import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["AGE_BIN"] = pd.cut(
        df["AGE"], bins=[0, 25, 35, 45, 55, 100], labels=False, include_lowest=True
    )
    df["PAY_AMT1_TO_LIMIT"] = df["PAY_AMT1"] / (df["LIMIT_BAL"] + 1e-6)
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["TOTAL_DELAYED"] = (df[pay_cols] > 0).sum(axis=1)
    return df


def main(raw_path: str, train_path: str, test_path: str, target_col: str):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    df = pd.read_csv(raw_path)
    df = feature_engineering(df)

    X_train, X_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )

    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)

    print("[build_features] Train shape:", X_train.shape)
    print("[build_features] Test shape:", X_test.shape)
    print("[build_features] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=str)
    parser.add_argument("train_path", type=str)
    parser.add_argument("test_path", type=str)
    parser.add_argument("--target", type=str, default="default.payment.next.month")
    args = parser.parse_args()
    main(args.raw_path, args.train_path, args.test_path, args.target)
