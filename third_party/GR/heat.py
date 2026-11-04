"""Load or synthesize user heat independently of request text."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ijson

from .scheduling import _integer

DEFAULT_HEAT_CURVES = Path(__file__).resolve().parent / "analysis" / "heat_curves.csv"


@dataclass
class HeatPopulation:
    weights: dict[int, float]
    metadata: dict

    @classmethod
    def from_curve(
        cls,
        path: str | Path = DEFAULT_HEAT_CURVES,
        *,
        dataset: str = "beauty",
        field: str = "interaction_count",
        num_users: int = 1000,
        seed: int = 42,
    ) -> HeatPopulation:
        """Approximate ranked heat using cumulative-traffic curve increments.

        Each synthetic user receives C(i/N) - C((i-1)/N), where C is the
        piecewise-linear curve with origin (0, 0). The seed shuffles these
        weights across synthetic IDs; it does not change the distribution.
        """
        _integer("num_users", num_users, 1)
        _integer("seed", seed)
        path = Path(path)
        raw = path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        required = {"dataset", "field", "user_fraction", "traffic_fraction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing curve columns {sorted(required)}")
        points = [(0.0, 0.0)]
        matched = 0
        previous_slope = math.inf
        for line, row in enumerate(reader, 2):
            if row["dataset"] != dataset or row["field"] != field:
                continue
            try:
                x, y = float(row["user_fraction"]), float(row["traffic_fraction"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("curve fractions must be finite")
                matched += 1
                if matched == 1 and (x, y) == (0.0, 0.0):
                    continue
                px, py = points[-1]
                if not (px < x <= 1 and py < y <= 1):
                    raise ValueError("curve fractions must increase strictly within [0, 1]")
                slope = (y - py) / (x - px)
                if slope > previous_slope and not math.isclose(
                    slope, previous_slope, rel_tol=1e-8, abs_tol=1e-10
                ):
                    raise ValueError("ranked traffic curve must be concave")
                points.append((x, y))
                previous_slope = slope
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
        if len(points) == 1:
            raise ValueError(f"{path}: no curve for {dataset}/{field}")
        if points[-1] != (1.0, 1.0):
            raise ValueError(f"{path}: curve must end at (1, 1)")

        ranked = []
        segment = 1
        previous = 0.0
        for rank in range(1, num_users + 1):
            fraction = rank / num_users
            while points[segment][0] < fraction:
                segment += 1
            left_x, left_y = points[segment - 1]
            right_x, right_y = points[segment]
            cumulative = left_y + (right_y - left_y) * ((fraction - left_x) / (right_x - left_x))
            weight = cumulative - previous
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("curve interpolation produced a nonpositive user weight")
            ranked.append(weight)
            previous = cumulative
        random.Random(f"{seed}:curve-population").shuffle(ranked)
        weights = dict(enumerate(ranked))
        return cls(
            weights,
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "dataset": dataset,
                "field": field,
                "synthetic": True,
                "method": "piecewise_linear_cumulative_increments",
                "curve_points": len(points) - 1,
                "selected_users": num_users,
                "seed": seed,
                "selection": "synthetic_population",
                "selected_weight_sum": math.fsum(weights.values()),
                "user_identity": "synthetic IDs 0..N-1; seeded shuffle of ranked weights",
            },
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        industrial_field: str | None = None,
        num_users: int = 1000,
        seed: int = 42,
    ) -> HeatPopulation:
        if num_users < 0:
            raise ValueError("num_users must be nonnegative (0 means all users)")
        rng = random.Random(f"{seed}:heat-population")
        if industrial_field is None:
            counts: Counter = Counter()
            with path.open("rb") as source:
                for _, pairs in ijson.kvitems(source, ""):
                    for uid, count in pairs:
                        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                            raise ValueError(f"{path}: invalid interaction count {count}")
                        counts[int(uid)] += count
            source_users = len(counts)
            users = sorted(counts)
            if num_users and num_users < source_users:
                users = rng.sample(users, num_users)
            rows = [(uid, float(counts[uid])) for uid in users]
        else:
            if industrial_field not in ("pv_share", "pv_int", "pv_scaled_1_100"):
                raise ValueError("unsupported industrial heat field")
            rows = []
            source_users = 0
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if not {"user_id", industrial_field}.issubset(reader.fieldnames or []):
                    raise ValueError(f"{path}: missing user_id or {industrial_field}")
                for line, row in enumerate(reader, 2):
                    try:
                        uid, weight = int(row["user_id"]), float(row[industrial_field])
                        if uid < 0 or not math.isfinite(weight) or weight <= 0:
                            raise ValueError("expected nonnegative ID and finite positive weight")
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{path}:{line}: {exc}") from exc
                    source_users += 1
                    if not num_users or len(rows) < num_users:
                        rows.append((uid, weight))
                    else:
                        index = rng.randrange(source_users)
                        if index < num_users:
                            rows[index] = (uid, weight)
        if not rows:
            raise ValueError("heat source contains no users")
        if num_users > source_users:
            raise ValueError(f"requested {num_users} users, source has only {source_users}")
        weights = dict(sorted(rows))
        if len(weights) != len(rows):
            raise ValueError("selected heat profiles contain duplicate user IDs")
        return cls(
            weights,
            {
                "path": str(path.resolve()),
                "field": industrial_field or "interaction_count",
                "source_users": source_users,
                "selected_users": len(weights),
                "seed": seed,
                "selection": "all"
                if len(weights) == source_users
                else "uniform_without_replacement",
                "selected_weight_sum": math.fsum(weights.values()),
                "user_identity": "original heat-source user ID; no text-user mapping",
            },
        )
