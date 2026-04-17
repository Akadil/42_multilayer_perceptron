from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the loss value given true labels and predicted outputs."""
        pass

    @abstractmethod
    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes the gradient of the loss with respect to the predicted outputs."""
        pass
