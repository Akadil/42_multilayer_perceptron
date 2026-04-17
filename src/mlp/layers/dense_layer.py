import numpy as np
from activations import Activation
from initializers import WeightsInitializer

from .utils.requires_compiled import requires_compiled


class DenseLayer:
    """A fully connected layer in a neural network.
    ===============================================================================================
    Formula:
        forward: output = activation_function(inputs @ weights + biases)
        backward: grad_input = (grad_output * activation_function.backward(z)) @ weights.T
    ===============================================================================================

    Attributes:
        num_neurons (int): The number of neurons in the layer.
        activation_function (Activation): The activation function to apply to the output of the layer.
        weight_initializer (WeightsInitializer): The initializer for the weights of the layer.
        weights (np.ndarray | None): The weights of the layer, shape (input_size, num_neurons).
        biases (np.ndarray | None): The biases of the layer, shape (num_neurons,).
    """

    def __init__(
        self,
        num_neurons: int,
        activation_function: Activation,
        weight_initializer: WeightsInitializer,
    ):
        self.num_neurons = num_neurons

        self.activation_function = activation_function
        self.weight_initializer = weight_initializer

        self.weights: np.ndarray | None = None  # shape (input_size, num_neurons)
        self.biases: np.ndarray | None = None  # shape (num_neurons,)

        # cache for backpropagation
        self._z_cache: np.ndarray | None = None  # shape (batch_size, num_neurons)
        self._input_cache: np.ndarray | None = None  # shape (batch_size, input_size)

        # gradients populated by backward(), consumed by optimizer
        self.grad_weights: np.ndarray | None = None  # shape (input_size, num_neurons)
        self.grad_biases: np.ndarray | None = None  # shape (num_neurons,)

    def compile(self, input_size: int):

        self.weights = self.weight_initializer(input_size, self.num_neurons)
        self.biases = np.zeros(self.num_neurons)

    @requires_compiled
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the layer.

        Args:
            inputs: shape(batch_size, input_size) The input data to the layer
        Returns:
            np.ndarray: shape(batch_size, num_neurons)
        Exceptions:
            ValueError: If the layer has not been compiled (weights and biases not initialized).
        """
        # Linear transformation
        self._input_cache = inputs
        self._z_cache = np.dot(inputs, self.weights) + self.biases

        return self.activation_function.forward(self._z_cache)

    @requires_compiled
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        Args:
            grad_output: shape (batch_size, num_neurons) — ∂L/∂output of this layer
        Returns:
            np.ndarray: shape (batch_size, input_size) — ∂L/∂input, passed to previous layer
        """
        grad_activation = grad_output * self.activation_function.backward(self._z_cache)

        self.grad_weights = np.dot(
            self._input_cache.T, grad_activation
        )  # (input_size, num_neurons)
        self.grad_biases = np.sum(grad_activation, axis=0)  # (num_neurons,)

        return np.dot(grad_activation, self.weights.T)  # (batch_size, input_size)

    def is_compiled(self) -> bool:
        """Checks if the layer has been compiled (weights and biases initialized).

        Returns:
            bool: True if the layer is compiled, False otherwise.
        """
        return self.weights is not None and self.biases is not None
