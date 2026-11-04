"""Synchronized wall-clock phase timing, enabled only for detailed profiling."""

import time
from collections.abc import Callable


class StageTimer:
    """Partition a region into consecutive, nonoverlapping wall-clock intervals.

    Each GPU boundary waits for completion before reading the clock. This includes
    Python preparation, memory transfers and GPU work, not just kernel duration.
    Synchronization changes overlap and adds overhead: use the disabled mode for
    normal throughput measurements. Timers never retain inputs or GPU tensors.
    """

    def __init__(
        self, synchronize: Callable[[], None], *, started_ns: int | None = None
    ):
        self.synchronize = synchronize
        self.started_ns = time.perf_counter_ns() if started_ns is None else started_ns
        self.last_ns = self.started_ns
        self.values: dict[str, float] = {}
        self.layers: list[dict] = []
        self.model: dict | None = None

    def mark(self, name: str, *, synchronize: bool = True) -> None:
        """Charge everything since the previous boundary to this phase."""
        if synchronize:
            self.synchronize()
        finished_ns = time.perf_counter_ns()
        self.values[name] = (finished_ns - self.last_ns) / 1_000_000
        self.last_ns = finished_ns

    def report(self) -> dict:
        """Return plain JSON data; total is measured at the last marked boundary."""
        result: dict = {
            **self.values,
            "total_ms": (self.last_ns - self.started_ns) / 1_000_000,
        }
        if self.layers:
            result["layers"] = self.layers
        if self.model is not None:
            result["model"] = self.model
        return result


def synchronize_model(model) -> None:
    """Complete model work without touching its returned hidden states.

    Production models implement BaseModel.synchronize. CPU-only test doubles
    without that hook have no asynchronous device work to wait for.
    """
    synchronize = getattr(model, "synchronize", None)
    if synchronize is not None:
        synchronize()
