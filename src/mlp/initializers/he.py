import numpy as np

from . import WeightsInitializer
    

class HeUniform(WeightsInitializer):
    def initialize(self, input_size: int, num_neurons: int) -> np.ndarray:
        limit = np.sqrt(6 / input_size)
        
        return np.random.uniform(-limit, limit, (input_size, num_neurons))