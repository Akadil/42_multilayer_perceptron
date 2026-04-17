from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    _registry: dict[str, type["ActivationFunction"]] = {}

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    @classmethod
    def from_str(cls, name: str) -> "ActivationFunction":
        if name not in cls._registry:
            raise ValueError(f"Unknown activation: '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name]()

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass
