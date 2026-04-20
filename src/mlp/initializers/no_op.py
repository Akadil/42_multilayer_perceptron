import numpy as np

from .base import WeightsInitializer


class NoOpInitializer(WeightsInitializer, name="no_op"):
    """Placeholder initializer used when loading a model from disk.
    Weights are restored directly and this initializer is never called.
    """
    def __str__(self):
        return "No-Op Initializer"

    def __repr__(self) -> str:
        return "NoOpInitializer()"
    
    def initialize(self, input_size: int, num_neurons: int) -> np.ndarray:
        raise RuntimeError("NoOpInitializer should not be called — restore weights from saved data.")
