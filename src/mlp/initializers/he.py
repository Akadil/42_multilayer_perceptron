import numpy as np

from . import WeightsInitializer


class HeUniform(WeightsInitializer, name="he_uniform"):
    def __str__(self):
        return "He Uniform Initializer"

    def __repr__(self) -> str:
        return "HeUniform()"

    def initialize(self, input_size: int, num_neurons: int) -> np.ndarray:
        limit = np.sqrt(6 / input_size)

        return np.random.uniform(-limit, limit, (input_size, num_neurons))
