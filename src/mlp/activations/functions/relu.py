from .. import ActivationFunction


class Relu(ActivationFunction, name="relu"):
    def __str__(self):
        return "ReLU Activation"
    
    def activate(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1.0 * (x > 0)
