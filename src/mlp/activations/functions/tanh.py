import numpy as np

class TanhActivation:
    @staticmethod
    def activate(x: np.ndarray) -> np.ndarray:
        """Applies the Tanh activation function element-wise."""
        return np.tanh(x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """Computes the derivative of the Tanh function."""
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2