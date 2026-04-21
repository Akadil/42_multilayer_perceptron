from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    _registry: dict[str, type["ActivationFunction"]] = {}

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls
        cls._name = name  # store the registry key on the class

    @property
    def name(self) -> str:
        return self._name

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
