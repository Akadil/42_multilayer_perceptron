import numpy as np

from . import Optimizer


class GradientDescent(Optimizer, name="gradient_descent"):
    """Implements the vanilla gradient descent optimization algorithm.

    This optimizer updates weights and biases by moving in the direction of the negative gradient
    of the loss function with respect to those parameters, scaled by a learning rate.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __str__(self):
        return f"Gradient Descent Optimizer (learning_rate={self.learning_rate})"

    def update(
        self,
        weights: np.ndarray,
        grad_weights: np.ndarray,
        biases: np.ndarray,
        grad_biases: np.ndarray,
    ):
        """Updates the weights and biases of the given layer using the computed gradients.

        Args:
            weights/biases: The current weights of the layer.
            grad_weights/grad_biases: The gradients of the weights and biases.
        """
        weights -= self.learning_rate * grad_weights
        biases -= self.learning_rate * grad_biases
