"""Lazy model selection; no GPU/model runtime is imported by Scheduler."""

from .base import BaseModel

MODEL_NAMES = ("test", "nsa_1b")


def resolve_model(name: str) -> type[BaseModel]:
    model_type: type[BaseModel]
    if name == "test":
        from cxl_recsys.models.simulation.model import TestModel

        model_type = TestModel
    elif name == "nsa_1b":
        from cxl_recsys.models.nsa import NSAModel

        model_type = NSAModel
    else:
        raise ValueError(f"Unknown model: {name}; expected one of {MODEL_NAMES}")
    model_type.ensure_available()
    return model_type
