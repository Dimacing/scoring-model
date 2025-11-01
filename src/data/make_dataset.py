import os
import argparse
import pandas as pd
import requests


UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/UCI_Credit_Card.csv"
)


def download_dataset(dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"[make_dataset] Downloading dataset from {UCI_URL} ...")
    resp = requests.get(UCI_URL)
    resp.raise_for_status()
    with open(dst_path, "wb") as f:
        f.write(resp.content)
    print(f"[make_dataset] Saved to {dst_path}")


def main(raw_path: str):
    if not os.path.exists(raw_path):
        download_dataset(raw_path)
    else:
        print(f"[make_dataset] File already exists: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"[make_dataset] Shape: {df.shape}")
    print("[make_dataset] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=str, help="Path to raw UCI_Credit_Card.csv")
    args = parser.parse_args()
    main(args.raw_path)
