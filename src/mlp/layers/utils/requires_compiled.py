import functools


def requires_compiled(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_compiled():
            raise ValueError("Layer must be compiled before calling this method.")
        return method(self, *args, **kwargs)

    return wrapper
