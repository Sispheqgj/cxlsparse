"""User sampling and request arrival schedules."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

GAP_SECONDS = (5, 15, 25, 35, 45, 55, 65)
GAP_WEIGHTS = (18, 12, 8, 5, 3, 2, 52)


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def index_letter(index: int) -> str:
    """Zero-based candidate index to A..Z, AA..AZ, ... ."""
    _integer("index", index)
    result = ""
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        result = chr(65 + digit) + result
    return result


@dataclass(frozen=True)
class ScheduleConfig:
    seed: int = 42
    qps: int = 100
    arrival: str = "poisson"
    sampling: str = "weighted"
    start_timestamp: float = 0.0

    def __post_init__(self):
        _integer("seed", self.seed)
        _integer("qps", self.qps, 1)
        if self.arrival not in ("poisson", "constant", "timeslot"):
            raise ValueError("arrival must be poisson, constant, or timeslot")
        if self.sampling not in ("weighted", "uniform", "sequential"):
            raise ValueError("sampling must be weighted, uniform, or sequential")
        if self.arrival == "timeslot" and self.sampling == "sequential":
            raise ValueError("timeslot requires weighted or uniform sampling")
        if not math.isfinite(self.start_timestamp) or self.start_timestamp < 0:
            raise ValueError("start_timestamp must be finite and nonnegative")


class _WeightedSampler:
    """Fenwick tree supporting weighted draw/removal in O(log users)."""

    def __init__(self, weights: list[float]):
        self.values = weights.copy()
        self.tree = [0.0] + weights.copy()
        for i in range(1, len(self.tree)):
            parent = i + (i & -i)
            if parent < len(self.tree):
                self.tree[parent] += self.tree[i]

    def set(self, index: int, value: float) -> None:
        delta = value - self.values[index]
        self.values[index] = value
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    def draw(self, rng: random.Random) -> int:
        total = 0.0
        i = len(self.values)
        while i:
            total += self.tree[i]
            i -= i & -i
        target = rng.random() * total
        index = 0
        step = 1 << (len(self.values).bit_length() - 1)
        while step:
            nxt = index + step
            if nxt < len(self.tree) and self.tree[nxt] <= target:
                target -= self.tree[nxt]
                index = nxt
            step >>= 1
        return min(index, len(self.values) - 1)


def _schedule(
    users: list[int], frequencies: dict[int, float], count: int, cfg: ScheduleConfig
) -> Iterator[tuple[int, float]]:
    rng = random.Random(cfg.seed)
    clock_rng = random.Random(f"{cfg.seed}:clock")
    gap_rng = random.Random(f"{cfg.seed}:gap")
    weights = [frequencies[uid] if cfg.sampling == "weighted" else 1 for uid in users]
    sampler = _WeightedSampler(weights)
    gaps = gap_rng.choices(GAP_SECONDS, weights=GAP_WEIGHTS, k=len(users))
    cooling: list[tuple[int, int]] = []
    active_count = len(users)
    slot = used = 0
    timestamp = cfg.start_timestamp
    for order in range(count):
        if cfg.arrival == "timeslot":
            if used == cfg.qps:
                slot += 1
                used = 0
            while True:
                while cooling and cooling[0][0] <= slot:
                    _, index = heapq.heappop(cooling)
                    sampler.set(index, weights[index])
                    active_count += 1
                if active_count:
                    break
                slot = cooling[0][0]
                used = 0
            index = sampler.draw(rng)
            timestamp = cfg.start_timestamp + slot
            sampler.set(index, 0)
            active_count -= 1
            heapq.heappush(cooling, (slot + gaps[index], index))
            used += 1
        else:
            index = order % len(users) if cfg.sampling == "sequential" else sampler.draw(rng)
            if cfg.arrival == "poisson":
                timestamp += clock_rng.expovariate(cfg.qps)
            else:
                timestamp = cfg.start_timestamp + order / cfg.qps
        yield users[index], timestamp
