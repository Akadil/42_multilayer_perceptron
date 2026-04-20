import numpy as np

from .. import ActivationFunction


class Sigmoid(ActivationFunction, name="sigmoid"):
    def __str__(self):
        return "Sigmoid Activation"

    def __repr__(self) -> str:
        return "Sigmoid()"

    def activate(self, x):
        """Applies the sigmoid activation function. Used in the forward pass."""
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """Computes the derivative of the sigmoid function. Used in the backward pass."""
        s = self.activate(x)
        return s * (1 - s)
