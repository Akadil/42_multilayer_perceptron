"""Loads a trained model and performs prediction on a dataset."""

import argparse

import numpy as np
import pandas as pd

from mlp.model import SequentialNeuralNetwork


def binary_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Computes the binary cross-entropy loss as defined in the project specification.

    E = -1/N * sum(y * log(p) + (1 - y) * log(1 - p))

    Args:
        y_true: shape (num_samples,) Binary true labels (0 or 1).
        y_pred_proba: shape (num_samples,) Predicted probabilities for the positive class.
    Returns:
        float: The average binary cross-entropy loss.
    """
    y_pred_proba = np.clip(y_pred_proba, 1e-9, 1 - 1e-9)  # prevent log(0)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


def main():
    parser = argparse.ArgumentParser(description="Predict using a trained MLP model.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--model", default="saved_model.npy", help="Path to the saved model.")
    args = parser.parse_args()

    model = SequentialNeuralNetwork.load(args.model)
    # print(f"Loaded model: {model}")

    df = pd.read_csv(args.dataset)
    y_true = df.diagnosis.to_numpy()
    X = df.drop(columns=["id", "diagnosis"]).astype(float).to_numpy()

    predictions = model.predict(X)
    proba = model.predict_proba(X)

    accuracy = np.mean(predictions == y_true)

    # binary cross-entropy: probability of the positive class (index 1)
    y_true_binary = (y_true == model.classes[1]).astype(int)
    loss = binary_cross_entropy(y_true_binary, proba[:, 1])

    print(f"loss    : {loss:.4f}")
    print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
