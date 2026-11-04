"""NSA-1B model and KV access backends."""


def __getattr__(name):
    if name == "NSAModel":
        from .model import NSAModel

        return NSAModel
    raise AttributeError(name)
