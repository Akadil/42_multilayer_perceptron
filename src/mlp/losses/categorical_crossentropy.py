import numpy as np

from . import LossFunction

EPSILON = 1e-15  # small constant to prevent log(0) in loss computation


class CategoricalCrossEntropyLoss(LossFunction):
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the categorical cross-entropy loss.

        Args:
            y_true: shape (batch_size, num_classes) One-hot encoded true labels.
            y_pred: shape (batch_size, num_classes) Predicted probabilities (output of softmax).
        Returns:
            float: The average loss over the batch.
        """
        # Clip y_pred to prevent log(0)
        y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)

        # Compute cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
        return loss

    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes the gradient of the categorical cross-entropy loss with respect to the predicted outputs.

        Args:
            y_true: shape (batch_size, num_classes) One-hot encoded true labels.
            y_pred: shape (batch_size, num_classes) Predicted probabilities (output of softmax).
        Returns:
            np.ndarray: shape (batch_size, num_classes) Gradient of the loss with respect to y_pred.
        """
        # Gradient of cross-entropy loss with softmax output is simply (y_pred - y_true)
        grad = (y_pred - y_true) / y_true.shape[0]  # Normalize by batch size
        return grad
