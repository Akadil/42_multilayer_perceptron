from collections.abc import Generator

import numpy as np

from mlp.layers.dense_layer import DenseLayer
from mlp.optimizers import GradientDescentOptimizer, Optimizer

SEED = 42


class SequentialNeuralNetwork:
    def __init__(self, layers: list[DenseLayer]):
        self.layers = layers
        self.optimizer: Optimizer | None = None  # set during compile()
        self.loss_function: LossFunction | None = None  # set during compile()

    def compile(self, input_size: int, optimizer: Optimizer | None = None, loss_function=None):
        self.optimizer = optimizer if optimizer is not None else GradientDescentOptimizer(0.01)
        self.loss_function = loss_function

        for layer in self.layers:
            layer.compile(input_size)
            input_size = layer.num_neurons

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=10, batch_size=32, validation_split=0.1):
        x_train, y_train, x_val, y_val = self._split_data(X, y, validation_split)

        for epoch in range(epochs):
            for x_batch, y_batch in self._create_batches(x_train, y_train, batch_size):
                # 1. Forward pass through the network to compute outputs.
                for layer in self.layers:
                    x_batch = layer.forward(x_batch)

                # 2. Compute loss and initial gradient based on the output of the last layer and y_batch.

                # 3. Backward pass through the network to compute gradients for all layers.
                for layer in reversed(self.layers):
                    grad_output = None  # This should be set to the appropriate value based on the loss gradient and next layer's gradients.
                    grad_input = layer.backward(grad_output)
                    grad_output = grad_input  # For the next layer in the backward pass

                # 4. Use the optimizer to update weights and biases of each layer based on computed gradients.
                # 5. Optionally, evaluate performance on the validation set at the end of each epoch.

                pass

        # For simplicity, this method is a stub. In a full implementation, it would:
        # 1. Split the data into training and validation sets based on validation_split.
        # 2. Loop over epochs and batches, performing forward and backward passes.

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Performs a forward pass through the network to generate predictions.

        Args:
            X: shape (num_samples, input_size) The input data for prediction.
        Returns:
            np.ndarray: shape (num_samples, output_size) The predicted outputs of the network.
        """
        # output = X
        # for layer in self.layers:
        #     output = layer.forward(output)
        # return output
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluates the model's performance on the given dataset.

        Args:
            X: shape (num_samples, input_size) The input data for evaluation.
            y: shape (num_samples, output_size) The true labels for the input data.
        Returns:
            float: The computed loss or accuracy of the model on the given dataset.
        """
        # For simplicity, this method is a stub. In a full implementation, it would:
        # 1. Use predict() to get the model's predictions on X.
        # 2. Compute and return a performance metric (e.g., loss or accuracy) comparing predictions
        # to y.
        pass

    def _split_data(
        self, X: np.ndarray, y: np.ndarray, split=0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the dataset into training and validation sets.

        Args:
            X: shape (num_samples, input_size) The input data to split.
            y: shape (num_samples, output_size) The labels corresponding to X.
            split: float The proportion of the dataset to include in the validation split.
        Returns:
            Tuple of (X_train, y_train, X_val, y_val) where:
                X_train: shape (num_samples * (1 - split), input_size) The training input data.
                y_train: shape (num_samples * (1 - split), output_size) The training labels.
                X_val: shape (num_samples * split, input_size) The validation input data.
                y_val: shape (num_samples * split, output_size) The validation labels.
        """
        rng = np.random.default_rng(SEED)
        indices = rng.permutation(len(X))
        split_idx = int(len(indices) * (1 - split))

        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _create_batches(
        self, X: np.ndarray, y: np.ndarray, batch_size: int
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Creates batches of data for training.

        Args:
            X: shape (num_samples, input_size) The input data to batch.
            y: shape (num_samples, output_size) The labels corresponding to X.
            batch_size: int The number of samples per batch.
        Returns:
            Generator of tuples (X_batch, y_batch) where:
                X_batch: shape (batch_size, input_size) A batch of input data.
                y_batch: shape (batch_size, output_size) The labels corresponding to X_batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            yield (X_batch, y_batch)
