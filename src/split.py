"""Splits a dataset CSV into training and validation sets."""

import argparse

import numpy as np
import pandas as pd

SEED = 42


def split_dataset(dataset_path: str, validation_split: float, output_dir: str) -> None:
    df = pd.read_csv(dataset_path)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(df))
    split_idx = int(len(indices) * (1 - validation_split))

    train_df = df.iloc[indices[:split_idx]]
    val_df = df.iloc[indices[split_idx:]]

    train_path = f"{output_dir}/data_training.csv"
    val_path = f"{output_dir}/data_validation.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"x_train shape : {train_df.shape}")
    print(f"x_valid shape : {val_df.shape}")
    print(f"> saved training set to '{train_path}'")
    print(f"> saved validation set to '{val_path}'")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--output_dir", default=".", help="Directory to save the split datasets.")
    args = parser.parse_args()

    split_dataset(args.dataset, args.validation_split, args.output_dir)


if __name__ == "__main__":
    main()