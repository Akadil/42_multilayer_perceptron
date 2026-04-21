"""
@TODO: (possibly) Add L2 regularization to the loss and gradients in backward().
@TODO: move the initializer logic to a separate method and call it from compile()
"""

import numpy as np

from ..activations import ActivationFunction
from ..initializers import HeUniform, WeightsInitializer
from .utils.requires_compiled import requires_compiled


class DenseLayer:
    """A fully connected layer in a neural network.
    ===============================================================================================
    Formula:
        forward: output = activation_function(inputs @ weights + biases)
        backward: grad_input = (grad_output * activation_function.backward(z)) @ weights.T
    ===============================================================================================

    Attributes:
        num_neurons - The number of neurons in the layer.
        activation_function - The activation function to apply after the linear transformation.
        weight_initializer - The initializer to use for the weights.

        weights - The weights of the layer
        biases - The biases of the layer
        grad_weights - Gradient of the loss with respect to weights
        grad_biases - Gradient of the loss with respect to biases

        _z_cache - Cache for the linear transformation output before activation
        _input_cache - Cache for the input to the layer
    """

    def __init__(
        self,
        num_neurons: int,
        activation_function: ActivationFunction,
        weight_initializer: WeightsInitializer | None = None,
    ):
        self.num_neurons = num_neurons
        self.activation_function = activation_function

        self.weight_initializer = weight_initializer or HeUniform()

        # weights and biases initialized in compile()
        self.weights: np.ndarray | None = None  # shape (input_size, num_neurons)
        self.biases: np.ndarray | None = None  # shape (num_neurons,)
        self.grad_weights: np.ndarray | None = None  # shape (input_size, num_neurons)
        self.grad_biases: np.ndarray | None = None  # shape (num_neurons,)

        # cache for backpropagation
        self._z_cache: np.ndarray | None = None  # shape (batch_size, num_neurons)
        self._input_cache: np.ndarray | None = None  # shape (batch_size, input_size)

    def __str__(self):
        return (
            f"DenseLayer(num_neurons={self.num_neurons}, "
            f"activation_function={self.activation_function})"
        )

    def __repr__(self):
        """Full state dump of the layer for debugging."""
        return (
            "DenseLayer(\n"
            f"  num_neurons={self.num_neurons},\n"
            f"  activation_function={self.activation_function!r},\n"
            f"  weight_initializer={self.weight_initializer!r},\n"
            f"  weights={self._format_array(self.weights)},\n"
            f"  biases={self._format_array(self.biases)},\n"
            f"  grad_weights={self._format_array(self.grad_weights)},\n"
            f"  grad_biases={self._format_array(self.grad_biases)},\n"
            f"  _z_cache={self._format_array(self._z_cache)},\n"
            f"  _input_cache={self._format_array(self._input_cache)}\n"
            ")"
        )

    @staticmethod
    def _format_array(value: np.ndarray | None) -> str:
        if value is None:
            return "None"
        return np.array2string(value, threshold=np.inf, separator=", ")

    def compile(self, input_size: int):
        """Initializes the weights and biases of the layer based on the input size and the number
        of neurons.

        Args:
            input_size (int): number of features from the previous layer.
        """
        self.weights = self.weight_initializer.initialize(input_size, self.num_neurons)
        self.biases = np.zeros(self.num_neurons)

    @requires_compiled
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer.

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

        return self.activation_function.activate(self._z_cache)

    @requires_compiled
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer.

        Args:
            grad_output - output gradient of the loss. shape (batch_size, num_neurons)
        Returns:
            np.ndarray - input gradient of the loss. shape (batch_size, input_size)
        Exceptions:
            ValueError: If the layer has not been compiled (weights and biases not initialized).
        """
        # gradient of the loss w.r.t. pre-activation output z
        # ∂L/∂z_i = ∂L/∂a_i * ∂a_i/∂z_i
        grad_activation = grad_output * self.activation_function.derivative(self._z_cache)

        # Compute gradients of the loss w.r.t. weights and biases
        # ∂L/∂w_i = ∂L/∂z_i * ∂z_i/∂w_i
        self.grad_weights = np.dot(
            self._input_cache.T, grad_activation
        )  # (input_size, num_neurons)
        self.grad_biases = np.sum(grad_activation, axis=0)  # (num_neurons,)

        # compute gradient of the loss w.r.t. inputs to pass to previous layer
        # ∂L/∂x_i = ∂L/∂z_i * ∂z_i/∂x_i
        grad_input = np.dot(grad_activation, self.weights.T)  # (batch_size, input_size)

        return grad_input

    def is_compiled(self) -> bool:
        """Checks if the layer has been compiled (weights and biases initialized).

        Returns:
            bool: True if the layer is compiled, False otherwise.
        """
        return self.weights is not None and self.biases is not None
