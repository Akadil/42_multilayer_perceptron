from abc import ABC, abstractmethod


class WeightsInitializer(ABC):
    def __call__(self, *args, **kwds):
        return self.initialize(*args, **kwds)

    @abstractmethod
    def initialize(self, shape):
        pass
