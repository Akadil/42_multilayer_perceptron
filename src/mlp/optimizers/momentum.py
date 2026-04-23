import numpy as np

from . import Optimizer


class Momentum(Optimizer, name="momentum"):
    """Implements SGD with momentum.

    Accumulates a velocity vector in directions of persistent gradient
    to accelerate learning and dampen oscillations.
    """

    def __init__(self, learning_rate: float, beta: float = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self._velocity_w: dict[int, np.ndarray] = {}
        self._velocity_b: dict[int, np.ndarray] = {}

    def __str__(self):
        return f"Momentum Optimizer (learning_rate={self.learning_rate}, beta={self.beta})"

    def __repr__(self):
        return f"Momentum(learning_rate={self.learning_rate!r}, beta={self.beta!r})"

    def update(
        self,
        weights: np.ndarray,
        grad_weights: np.ndarray,
        biases: np.ndarray,
        grad_biases: np.ndarray,
    ):
        key = id(weights)

        if key not in self._velocity_w:
            self._velocity_w[key] = np.zeros_like(weights)
            self._velocity_b[key] = np.zeros_like(biases)

        self._velocity_w[key] = (
            self.beta * self._velocity_w[key] + self.learning_rate * grad_weights
        )
        self._velocity_b[key] = self.beta * self._velocity_b[key] + self.learning_rate * grad_biases

        weights -= self._velocity_w[key]
        biases -= self._velocity_b[key]
