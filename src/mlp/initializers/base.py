from abc import ABC, abstractmethod

import numpy as np


class WeightsInitializer(ABC):
    _registry: dict[str, type["WeightsInitializer"]] = {}

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    @classmethod
    def from_str(cls, name: str) -> "WeightsInitializer":
        if name not in cls._registry:
            raise ValueError(f"Unknown initializer: '{name}'. Available: {list(cls._registry)}")
        return cls._registry[name]()

    @abstractmethod
    def initialize(self, input, output) -> np.ndarray:
        pass
