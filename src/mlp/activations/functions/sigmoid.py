import numpy as np

from .. import Activation


class Sigmoid(Activation):
    def activate(self, x):
        """Applies the sigmoid activation function. Used in the forward pass."""
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """Computes the derivative of the sigmoid function. Used in the backward pass."""
        s = self.forward(x)
        return s * (1 - s)
