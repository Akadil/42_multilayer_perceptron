import numpy as np

from .. import ActivationFunction


class Tanh(ActivationFunction, name="tanh"):
    def __str__(self):
        return "Tanh Activation"

    def activate(self, x: np.ndarray) -> np.ndarray:
        """Applies the Tanh activation function element-wise."""
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Computes the derivative of the Tanh function."""
        tanh_x = np.tanh(x)
        return 1 - tanh_x**2
