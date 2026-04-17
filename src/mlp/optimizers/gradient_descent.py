from . import Optimizer


class GradientDescent(Optimizer):
    """Implements the vanilla gradient descent optimization algorithm.

    This optimizer updates weights and biases by moving in the direction of the negative gradient
    of the loss function with respect to those parameters, scaled by a learning rate.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, layer):
        """Updates the weights and biases of the given layer using the computed gradients.

        Args:
            layer: The layer whose parameters are to be updated. Must have grad_weights and grad_biases attributes.


        Exceptions:
            ValueError: If the layer does not have grad_weights or grad_biases computed.
        """
        if layer.grad_weights is None or layer.grad_biases is None:
            raise ValueError(
                "Gradients for weights and biases must be computed before calling step()."
            )

        # Update weights and biases using the gradients and learning rate
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.biases -= self.learning_rate * layer.grad_biases
