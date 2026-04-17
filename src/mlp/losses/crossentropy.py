import numpy as np


class CrossEntropyWithSoftmax:
    def __init__(self):
        self._output_cache: np.ndarray | None = None  # Cache for the output of the softmax function

    def compute_loss(self, y_true: np.ndarray, z_pred: np.ndarray) -> float:
        """Computes the categorical cross-entropy loss.

        Args:
            y_true: shape (batch_size, num_classes) One-hot encoded true labels.
            z_pred: shape (batch_size, num_classes) Logits (pre-softmax values).
        Returns:
            float: The average loss over the batch.
        """
        # Apply softmax to get predicted probabilities
        y_pred = self.softmax_activation(z_pred)
        self._output_cache = y_pred  # Cache the softmax output for use in gradient computation

        # Compute cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def compute_gradient(self, y_true: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
        """Computes the gradient of the loss with respect to the predicted outputs.

        Formulas:
        - ∂L/∂z_pred = (y_pred - y_true) / batch_size

        Args:
            y_true: One-hot encoded true labels. shape (batch_size, num_classes)
            z_pred: Logits (pre-softmax values). shape (batch_size, num_classes)
        Returns:
            np.ndarray: Gradient of the loss with respect to z_pred. shape (batch_size, num_classes)
        """
        # Apply softmax to get predicted probabilities
        y_pred = (
            self._output_cache
            if self._output_cache is not None
            else self.softmax_activation(z_pred)
        )

        # Gradient of cross-entropy loss with softmax output is simply (y_pred - y_true)
        grad = (y_pred - y_true) / y_true.shape[0]  # Normalize by batch size
        return grad

    def softmax_activation(self, z: np.ndarray) -> np.ndarray:
        """Applies the softmax activation function to the input.

        Args:
            z: shape (batch_size, num_classes) The input to the softmax function (logits).
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
