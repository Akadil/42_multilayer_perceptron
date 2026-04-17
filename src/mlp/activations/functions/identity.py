from .. import Activation


class Identity(Activation, name="identity"):
    def __str__(self):
        return "Identity Activation"

    def activate(self, x):
        return x

    def derivative(self, _):
        return 1
