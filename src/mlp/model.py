from collections.abc import Generator

import numpy as np

from .activations.base import ActivationFunction
from .initializers import NoOpInitializer
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

    def __str__(self):
        return f"SequentialNeuralNetwork(layers={self.layers}, optimizer={self.optimizer})"

    def __repr__(self) -> str:
        layers_repr = "[\n" + ",\n".join(repr(layer) for layer in self.layers) + "\n]"
        return (
            "SequentialNeuralNetwork(\n"
            f"  layers={layers_repr},\n"
            f"  optimizer={self.optimizer!r},\n"
            f"  loss_function={self.loss_function!r},\n"
            f"  mean={self._format_array(self.mean)},\n"
            f"  std={self._format_array(self.std)},\n"
            f"  classes={self._format_array(self.classes)},\n"
            f"  history={self.history!r}\n"
            ")"
        )

    @staticmethod
    def _format_array(value: np.ndarray | None) -> str:
        if value is None:
            return "None"
        return np.array2string(value, threshold=np.inf, separator=", ")

    @classmethod
    def load(cls, path: str) -> "SequentialNeuralNetwork":
        """Loads a trained model from disk.

        Args:
            path: File path to the saved model (e.g. 'saved_model.npy').
        Returns:
            SequentialNeuralNetwork: The reconstructed model ready for inference.
        """
        data = np.load(path, allow_pickle=True).item()

        num_layers = data["num_layers"]
        layers = []

        for i in range(num_layers):
            activation_name = data[f"layer_{i}_activation"]
            activation_function = ActivationFunction.from_str(activation_name)
            num_neurons = int(data[f"layer_{i}_num_neurons"])

            layer = DenseLayer(
                num_neurons=num_neurons,
                activation_function=activation_function,
                weight_initializer=NoOpInitializer(),  # weights restored below
            )
            # restore weights directly — bypasses compile()
            layer.weights = data[f"layer_{i}_weights"]
            layer.biases = data[f"layer_{i}_biases"]
            layers.append(layer)

        model = cls(layers)
        model.mean = data["mean"]
        model.std = data["std"]
        model.classes = data["classes"]

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

    def save(self, path: str) -> None:
        """Serializes the trained model to disk.

        Args:
            path: File path to save the model (e.g. 'saved_model.npy').
        """
        if not self.is_trained():
            raise RuntimeError("Cannot save an untrained model.")

        data = {
            "mean": self.mean,
            "std": self.std,
            "classes": self.classes,
            "num_layers": len(self.layers),
        }

        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_weights"] = layer.weights
            data[f"layer_{i}_biases"] = layer.biases
            data[f"layer_{i}_activation"] = layer.activation_function.name
            data[f"layer_{i}_num_neurons"] = layer.num_neurons

        np.save(path, data)

    def is_trained(self) -> bool:
        """Checks if the model has been trained"""
        return self.mean is not None and self.std is not None and self.classes is not None

    def _split_data(
        self, X: np.ndarray, y: np.ndarray, split: float = 0.1
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
