import functools


def requires_training(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_trained():
            raise ValueError("Layer must be trained before calling this method.")
        return method(self, *args, **kwargs)

    return wrapper
