"""Trains a multilayer perceptron on a dataset and saves the model to disk."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlp.layers.dense_layer import DenseLayer
from mlp.losses.crossentropy import CrossEntropyWithSoftmax
from mlp.network import SequentialNeuralNetwork
from mlp.optimizers import GradientDescentOptimizer
from mlp.activations import Sigmoid, Softmax
from mlp.initializers import HeUniform, RandomNormal


ACTIVATION_REGISTRY = {
    "sigmoid": Sigmoid,
    "softmax": Softmax,
}

INITIALIZER_REGISTRY = {
    "heUniform": HeUniform,
    "randomNormal": RandomNormal,
}


def build_network(layer_sizes: list[int], input_size: int, output_size: int) -> list[DenseLayer]:
    layers = []
    sizes = [input_size] + layer_sizes + [output_size]

    for i in range(len(sizes) - 1):
        is_output = i == len(sizes) - 2
        activation = Softmax() if is_output else Sigmoid()
        initializer = HeUniform()
        layer = DenseLayer(
            num_neurons=sizes[i + 1],
            activation_function=activation,
            weight_initializer=initializer,
        )
        layers.append(layer)

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
    parser = argparse.ArgumentParser(description="Train a multilayer perceptron.")
    parser.add_argument("--dataset", required=True, help="Path to the training CSV file.")
    parser.add_argument("--layer", nargs="+", type=int, default=[24, 24], help="Hidden layer sizes.")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0314)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--output", default="saved_model.npy", help="Path to save the model.")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    # assumes last column is the target, rest are features
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values

    input_size = X.shape[1]
    output_size = len(np.unique(y))

    layers = build_network(args.layer, input_size, output_size)
    model = SequentialNeuralNetwork(layers)
    model.compile(
        input_size=input_size,
        optimizer=GradientDescentOptimizer(args.learning_rate),
    )

    print(f"x_train shape : {X.shape}")
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.validation_split)

    model.save(args.output)
    print(f"> saving model '{args.output}' to disk...")

    plot_learning_curves(model.history)


if __name__ == "__main__":
    main()