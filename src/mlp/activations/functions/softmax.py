# import numpy as np

# class SoftmaxActivation:
#     @staticmethod
#     def activate(x: np.ndarray) -> np.ndarray:
#         """Applies the softmax activation function to the input array.

#         formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j in all classes

#         The subtraction of np.max(x, axis=1, keepdims=True) from
#                     x - np.max(x, axis=1, keepdims=True)
#         is used for numerical stability to prevent overflow when computing exp(x).

#         Args:
#             x: shape (batch_size, num_classes) The input to the softmax function.
#         Returns:
#             np.ndarray: shape (batch_size, num_classes) The output probabilities
#         """
#         exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#     @staticmethod
#     def derivative(x):
#         s = SoftmaxActivation.activate(x)
#         return s * (1 - s)

