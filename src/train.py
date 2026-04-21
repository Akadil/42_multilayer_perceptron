"""Trains a multilayer perceptron on a dataset and saves the model to disk."""

import matplotlib.pyplot as plt
import pandas as pd

from mlp.activations import Identity, Sigmoid
from mlp.layers.dense_layer import DenseLayer
from mlp.model import SequentialNeuralNetwork
from mlp.optimizers import GradientDescentOptimizer
from utils import parse_arguments


def build_network(hidden_layers: list[int]) -> list[DenseLayer]:
    layers = [
        DenseLayer(
            num_neurons=layer_size,
            activation_function=Sigmoid(),
        )
        for layer_size in hidden_layers
    ] + [
        DenseLayer(
            num_neurons=2,
            activation_function=Identity(),  # Using identity for output layer for simplicity
        )
    ]
    return layers


def plot_learning_curves(history: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["loss"], label="training loss")
    ax1.plot(history["val_loss"], linestyle="--", label="validation loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["accuracy"], label="training acc")
    ax2.plot(history["val_accuracy"], linestyle="--", label="validation acc")
    ax2.set_title("Learning Curves")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()

    # Load dataset
    df = pd.read_csv(args.dataset)

    y = df.diagnosis.to_numpy()
    X = df.drop(columns=["id", "diagnosis"]).astype(float).to_numpy()

    # Build the model
    model = SequentialNeuralNetwork(build_network(args.layer))

    model.compile(
        input_size=X.shape[1],
        optimizer=GradientDescentOptimizer(args.learning_rate),
    )

    # Train the model
    print(f"x_train shape : {X.shape}")
    model.fit(
        X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split
    )

    model.save(args.output)
    print(f"> saving model '{args.output}' to disk...")

    plot_learning_curves(model.history)


if __name__ == "__main__":
    main()
