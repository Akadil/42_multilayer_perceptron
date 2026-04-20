from collections.abc import Generator

import numpy as np

from .activations.base import ActivationFunction
from .layers.dense_layer import DenseLayer
from .losses.crossentropy import CrossEntropyWithSoftmax
from .optimizers import GradientDescentOptimizer, Optimizer
from .utils import requires_training

SEED = 42


class SequentialNeuralNetwork:
    """A simple feedforward neural network composed of a sequence of layers.
    ===============================================================================================
    Formula:

    Important:
        - the network assumes Loss function is CrossEntropyLoss and the output layer uses Softmax
            activation. This is a common setup for classification tasks.

    Attributes:
    - layers: A list of DenseLayer instances that make up the network.
    - optimizer: An instance of an Optimizer used to update the weights during training.
    - loss_function: the cross-entropy loss function with softmax activation.
    ===============================================================================================
    """

    def __init__(self, layers: list[DenseLayer]):
        if len(layers) == 0:
            raise ValueError("The network must have at least one layer.")
        if layers[-1].activation_function != "identity":
            raise ValueError(
                "The output layer must use the identity activation function when using "
                "CrossEntropyWithSoftmax loss."
            )

        self.layers = layers
        self.optimizer: Optimizer | None = None  # set during compile()
        self.loss_function = CrossEntropyWithSoftmax()  # CrossEntropyLossWithSofmax
        self.mean: np.ndarray | None = None  # mean of the training data for normalization
        self.std: np.ndarray | None = None  # std of the training data for normalization
        self.history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }  # to track training and validation loss over epochs
        self.classes: np.ndarray | None = None

    # load the parameters of the model from a file
    @classmethod
    def load(cls, file_path: str) -> "SequentialNeuralNetwork":
        """Loads the model parameters from a file and creates a SequentialNeuralNetwork instance.

        Args:
            file_path (str): The path to the file containing the model parameters.
        Returns:
            SequentialNeuralNetwork: An instance of SequentialNeuralNetwork with the loaded
            parameters.
        """
        data = np.load(file_path, allow_pickle=True)
        layers_data = data["layers"]
        mean = data["mean"]
        std = data["std"]
        classes = data["classes"]

        layers = []
        for weights, biases, activation_name in layers_data:
            activation_function = ActivationFunction.from_str(activation_name)
            layer = DenseLayer(
                num_neurons=weights.shape[1], activation_function=activation_function
            )
            layer.weights = weights
            layer.biases = biases
            layers.append(layer)

        model = cls(layers)
        model.mean = mean
        model.std = std
        model.classes = classes

        return model

    def compile(self, input_size: int, optimizer: Optimizer | None = None):
        self.optimizer = optimizer if optimizer is not None else GradientDescentOptimizer(0.01)

        for layer in self.layers:
            layer.compile(input_size)
            input_size = layer.num_neurons

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=10, batch_size=32, validation_split=0.1):
        """Trains the neural network on the provided dataset.

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): _description_
            epochs (int, optional): _description_. Defaults to 10.
            batch_size (int, optional): _description_. Defaults to 32.
            validation_split (float, optional): _description_. Defaults to 0.1.
        """
        self.classes = np.unique(y)
        y = np.searchsorted(self.classes, y)  # Convert labels to integer indices
        x_train, y_train, x_val, y_val = self._split_data(X, y, validation_split)

        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)

        # Normalize the input data (zero mean and unit variance) for better training performance.
        x_train = self._normalize(x_train)
        x_val = self._normalize(x_val)
        y_train = np.eye(len(self.classes))[y_train]
        y_val = np.eye(len(self.classes))[y_val]

        for epoch in range(epochs):
            for x_batch, y_batch in self._create_batches(x_train, y_train, batch_size):
                # calculate the learning steps (parameters gradients of the loss)
                probs = self._forward(x_batch)
                grad = self.loss_function.compute_gradient(y_batch, probs)
                self._backward(grad)

                # update the parameters of the network
                for layer in self.layers:
                    self.optimizer.update(
                        layer.weights, layer.grad_weights, layer.biases, layer.grad_biases
                    )

            self._evaluate(epoch, epochs, x_train, y_train, x_val, y_val)

    @requires_training
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output (class labels) for the given input data.

        Args:
            X: shape (num_samples, input_size) The input data for prediction.
        Returns:
            np.ndarray: The predicted outputs of the network. shape (num_samples,).
            example: ['class1', 'class2', 'class1', ...]
        """
        probs = self._forward(self._normalize(X))

        return self.classes[np.argmax(probs, axis=1)]

    @requires_training
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output probabilities for the given input data."""
        return self._softmax(self._forward(self._normalize(X)))

    def save(self, file_path: str) -> None:
        """Saves the model parameters to a file."""
        np.savez(
            file_path,
            layers=[
                (layer.weights, layer.biases, layer.activation_function.name)
                for layer in self.layers
            ],
            mean=self.mean,
            std=self.std,
            classes=self.classes,
        )

    def is_trained(self) -> bool:
        """Checks if the model has been trained"""
        return self.mean is not None and self.std is not None and self.classes is not None

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

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Performs a forward pass through the network to compute the output."""
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Performs a backward pass through the network to compute gradients."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output

    def _evaluate(
        self,
        epoch,
        epoch_total,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Evaluates the model on the provided dataset.

        Args:
            epoch: int The current epoch number.
            epoch_total: int The total number of epochs for training.
            x_train: shape (num_samples, input_size) The training input data.
            y_train: shape (num_samples, output_size) The training labels.
            x_val: shape (num_samples, input_size) The validation input data.
            y_val: shape (num_samples, output_size) The validation labels.
        """
        # loss and accuracy for training data
        probs = self._forward(x_train)
        self.history["loss"].append(self.loss_function.compute_loss(y_train, probs))
        self.history["accuracy"].append(
            np.mean(np.argmax(probs, axis=1) == np.argmax(y_train, axis=1))
        )

        # validation
        probs = self._forward(x_val)
        self.history["val_loss"].append(self.loss_function.compute_loss(y_val, probs))
        self.history["val_accuracy"].append(
            np.mean(np.argmax(probs, axis=1) == np.argmax(y_val, axis=1))
        )
        # print: epoch 01/70 - loss: 0.6882 - val_loss: 0.6788
        print(
            f"epoch {epoch:02d}/{epoch_total:02d} - "
            f"loss: {self.history['loss'][-1]:.4f} - "
            f"val_loss: {self.history['val_loss'][-1]:.4f}"
        )

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

    def _normalize(self, input_data: np.ndarray) -> np.ndarray:
        """Normalizes the data with the training data statistics."""
        if self.mean is None or self.std is None:
            raise ValueError("Mean and standard deviation must be computed from training data")

        std = np.where(self.std == 0, 1, self.std)  # Prevent division by zero
        return (input_data - self.mean) / std

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Applies the softmax activation function to the input.

        Args:
            z: shape (batch_size, num_classes) The input to the softmax function (logits).
        Returns:
            np.ndarray: shape (batch_size, num_classes) The output probabilities after applying
            softmax.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
