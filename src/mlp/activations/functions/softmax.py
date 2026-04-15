import numpy as np

class SoftmaxActivation:
    @staticmethod
    def activate(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # @staticmethod
    # def derivative(x):
    #     s = SoftmaxActivation.activate(x)
    #     return s * (1 - s)
    
