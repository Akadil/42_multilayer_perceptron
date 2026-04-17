from .. import Activation


class Identity(Activation):
    def forward(self, x):
        return x

    def backward(self, _):
        return 1
