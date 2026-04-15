class Relu:
    def __call__(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)