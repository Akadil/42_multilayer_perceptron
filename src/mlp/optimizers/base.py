from abc import ABC, abstractmethod


class Optimizer(ABC):
    _registry: dict[str, type["Optimizer"]] = {}

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    @classmethod
    def from_str(cls, name: str) -> "Optimizer":
        if name not in cls._registry:
            raise ValueError(f"Unknown optimizer: '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name]()

    @abstractmethod
    def update(self, layer):
        """Updates the weights and biases of the given layer based on its stored gradients."""
        pass
