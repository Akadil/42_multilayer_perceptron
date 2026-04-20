from .. import ActivationFunction


class Identity(ActivationFunction, name="identity"):
    def __str__(self):
        return "Identity Activation"

    def __repr__(self) -> str:
        return "Identity()"

    def activate(self, x):
        return x

    def derivative(self, _):
        return 1
