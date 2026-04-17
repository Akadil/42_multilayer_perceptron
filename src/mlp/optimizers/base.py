from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(self, layer):
        """Updates the weights and biases of the given layer based on its stored gradients."""
        pass
