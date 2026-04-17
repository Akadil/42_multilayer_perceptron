from .. import Activation


class Identity(Activation):
    def activate(self, x):
        return x

    def derivative(self, _):
        return 1
