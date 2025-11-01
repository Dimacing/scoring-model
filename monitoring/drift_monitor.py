import argparse
import numpy as np
import pandas as pd
from joblib import load


def psi(expected, actual, buckets=10):
    def scale_range(i):
        return (i - i.min()) / (i.max() - i.min() + 1e-9)

    expected_percents = np.histogram(scale_range(expected), bins=buckets)[0] / len(
        expected
    )
    actual_percents = np.histogram(scale_range(actual), bins=buckets)[0] / len(actual)
    psi_val = np.sum(
        (expected_percents - actual_percents)
        * np.log((expected_percents + 1e-9) / (actual_percents + 1e-9))
    )
    return psi_val


def main(train_path: str, new_data_path: str, model_path: str):
    train_df = pd.read_csv(train_path)
    new_df = pd.read_csv(new_data_path)

    model = load(model_path)

    target = "default.payment.next.month"
    X_train = train_df.drop(columns=[target])
    X_new = new_df.drop(columns=[target], errors="ignore")

    train_proba = model.predict_proba(X_train)[:, 1]
    new_proba = model.predict_proba(X_new)[:, 1]

    psi_proba = psi(train_proba, new_proba)
    psi_limit = psi(train_df["LIMIT_BAL"], new_df["LIMIT_BAL"])

    print(f"[drift_monitor] PSI(proba): {psi_proba:.4f}")
    print(f"[drift_monitor] PSI(LIMIT_BAL): {psi_limit:.4f}")

    if psi_proba > 0.2 or psi_limit > 0.2:
        print("[drift_monitor] Drift detected!")
    else:
        print("[drift_monitor] No significant drift.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--new", default="data/processed/test.csv")
    parser.add_argument("--model", default="models/credit_default_model.pkl")
    args = parser.parse_args()
    main(args.train, args.new, args.model)
