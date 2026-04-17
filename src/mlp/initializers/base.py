from abc import ABC, abstractmethod

import numpy as np


class WeightsInitializer(ABC):
    @abstractmethod
    def initialize(self, input, output) -> np.ndarray:
        pass
