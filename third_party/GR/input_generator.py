"""Readable, exact-token-budget requests with independent user heat and arrival time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from .dataset import DEFAULT_DATA_ROOT, load_titles
from .heat import DEFAULT_HEAT_CURVES, HeatPopulation
from .scheduling import ScheduleConfig, _integer, _schedule, index_letter

DEFAULT_USER_LENGTHS = (4096, 16384, 65536, 262144, 1048576)
DEFAULT_ITEM_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_MAX_INPUT_TOKENS = max(DEFAULT_USER_LENGTHS) + max(DEFAULT_ITEM_LENGTHS)

DEFAULT_TOKENIZER = Path("/mnt/nfs/share/models/DeepSeek-V3.2/tokenizer.json")

INSTRUCTION = (
    "Recommend one item from the candidate pool based on the user history. Output its index letter."
)
PREFIX = f"<｜begin▁of▁sentence｜>{INSTRUCTION}<｜User｜>User history:\n"
ENDING = "<｜Assistant｜></think>"
ATTRIBUTES = (
    "easy to clean",
    "comfortable to use",
    "compact and lightweight",
    "suitable for daily use",
    "simple to store",
    "designed for travel",
)
PRODUCTS = (
    "travel bag",
    "desk lamp",
    "water bottle",
    "running shoes",
    "wireless headphones",
    "coffee maker",
    "notebook",
    "storage box",
    "cotton shirt",
    "face moisturizer",
)
COLORS = ("blue", "green", "black", "white", "silver", "red")
# Whole sentences used only to close the small remainder after complete records.
TAILS = (
    "Looks good.\n",
    "Quality is good.\n",
    "The design is practical.\n",
    "The finish is smooth.\n",
    "The size is convenient.\n",
    "Checked.\n",
)


@dataclass(frozen=True)
class TextConfig:
    user_lengths: tuple[int, ...] = DEFAULT_USER_LENGTHS
    user_probabilities: tuple[float, ...] = (1 / len(DEFAULT_USER_LENGTHS),) * len(
        DEFAULT_USER_LENGTHS
    )
    item_lengths: tuple[int, ...] = DEFAULT_ITEM_LENGTHS
    item_probabilities: tuple[float, ...] = (1 / len(DEFAULT_ITEM_LENGTHS),) * len(
        DEFAULT_ITEM_LENGTHS
    )
    candidate_count: int = 20
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    history_cache_users: int = 8
    prompt_format: str = "deepseek"

    def __post_init__(self):
        if self.prompt_format not in ("deepseek", "nosa", "nsa"):
            raise ValueError("prompt_format must be deepseek, nosa or nsa")
        for name in ("user", "item"):
            lengths = getattr(self, name + "_lengths")
            probabilities = getattr(self, name + "_probabilities")
            if not lengths or len(lengths) != len(probabilities):
                raise ValueError(
                    f"{name}: lengths and probabilities must be nonempty and equally sized"
                )
            for length in lengths:
                _integer(name + "_length", length, 1)
            if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(
                sum(probabilities), 1
            ):
                raise ValueError(f"{name}: probabilities must be finite, nonnegative and sum to 1")
        _integer("candidate_count", self.candidate_count, 1)
        _integer("max_input_tokens", self.max_input_tokens, 1)
        _integer("history_cache_users", self.history_cache_users)
        if max(self.user_lengths) + max(self.item_lengths) > self.max_input_tokens:
            raise ValueError("user + item length exceeds max_input_tokens")


class InputGenerator:
    def __init__(
        self,
        population: HeatPopulation,
        tokenizer: Tokenizer,
        *,
        text_config: TextConfig | None = None,
        schedule_config: ScheduleConfig | None = None,
        titles: dict[int, str] | None = None,
    ):
        self.population = population
        self.tokenizer = tokenizer
        tokenizer.no_padding()
        tokenizer.no_truncation()
        self.text_config = text_config or TextConfig()
        self.schedule_config = schedule_config or ScheduleConfig(sampling="weighted")
        self.seed = self.schedule_config.seed
        self.users = sorted(population.weights)
        if not self.users:
            raise ValueError("population must not be empty")
        self.titles = titles
        if titles is not None and not titles:
            raise ValueError("text catalog is empty")
        self.catalog = sorted(titles) if titles is not None else range(1_000_000)
        if len(self.catalog) < self.text_config.candidate_count:
            raise ValueError("candidate_count exceeds catalog size")
        self.prefix = PREFIX
        self.ending = ENDING
        self.history_ending = ""
        self.tails = TAILS
        specials = ("<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>")
        if self.text_config.prompt_format == "nosa":
            # NOSA prepends a SentencePiece space to independently encoded
            # sections. Its existing special delimiter prevents cross-section
            # merges and makes full-prompt encoding exactly additive.
            self.prefix = f"<s>{INSTRUCTION} User history:<|im_end|>"
            self.ending = "\nAnswer:"
            self.history_ending = "<|im_end|>"
            # NOSA's existing tails have 3/5/6-token increments before Answer:.
            # Complete 2/4-token sentences cover the small remaining budgets.
            self.tails = TAILS + ("Noted.\n", " Fine.")
            specials = ("<s>", "<|im_end|>")
        elif self.text_config.prompt_format == "nsa":
            # Llama tokenizer boundaries keep history and candidates exactly
            # additive, while preserving the common single-letter answer prompt.
            self.prefix = f"<|begin_of_text|>{INSTRUCTION} User history:<|eot_id|>"
            self.ending = "\nAnswer:\n"
            self.history_ending = "<|eot_id|>"
            specials = ("<|begin_of_text|>", "<|eot_id|>")
        for special in specials:
            if tokenizer.token_to_id(special) is None:
                raise ValueError(f"tokenizer lacks {self.text_config.prompt_format} token {special}")
        self.prefix_ids = self.encode(self.prefix)
        self._title = lru_cache(maxsize=16384)(self._make_title)
        self._history = lru_cache(maxsize=self.text_config.history_cache_users)(self._make_history)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _make_title(self, item: int) -> str:
        if self.titles is not None:
            words = self.titles[item].split()[:24]
            while words and len(self.encode(" ".join(words))) > 24:
                words.pop()
            return " ".join(words) if words else f"Catalog product P{item}"
        rng = random.Random(f"{self.seed}:product:{item}")
        return f"{rng.choice(COLORS)} {rng.choice(PRODUCTS)}"

    def lengths_for_user(self, uid: int) -> tuple[int, int]:
        cfg = self.text_config
        # Independent streams avoid accidental coupling with user heat and scheduling.
        # Keep the original RNG labels so renaming the options does not change draws.
        h = random.Random(f"{self.seed}:history-length:{uid}").choices(
            cfg.user_lengths, weights=cfg.user_probabilities, k=1
        )[0]
        n = random.Random(f"{self.seed}:non-history-length:{uid}").choices(
            cfg.item_lengths, weights=cfg.item_probabilities, k=1
        )[0]
        return h, n

    def _fit(self, heading: str, records, target: int, ending: str = "") -> str:
        """Keep complete records; close the residual budget with short sentences.

        Full-block tokenization verifies the budget. No arbitrary ID padding or
        partially decoded token prefix is used to fill the token budget.
        """
        base = len(self.encode(heading + ending))
        if base > target:
            raise ValueError(f"mandatory text uses {base} tokens, budget is {target}")
        lines = []
        estimate = base
        while estimate < target + 64:
            line = next(records)
            lines.append(line)
            estimate += max(1, len(self.encode(line)))
        # Find the longest record prefix fitting the block budget.
        low, high = 0, len(lines)
        while low < high:
            mid = (low + high + 1) // 2
            if len(self.encode(heading + "".join(lines[:mid]) + ending)) <= target:
                low = mid
            else:
                high = mid - 1
        for keep in range(low, max(-1, low - 4), -1):
            body = heading + "".join(lines[:keep])
            remainder = target - len(self.encode(body + ending))
            # Measure contextual increments rather than assume tokenizer additivity.
            increments = [
                len(self.encode(body + tail + ending)) - (target - remainder) for tail in self.tails
            ]
            solutions: dict[int, list[str]] = {0: []}
            for total in range(1, remainder + 1):
                for tail, inc in zip(self.tails, increments, strict=True):
                    if inc > 0 and total - inc in solutions:
                        solutions[total] = solutions[total - inc] + [tail]
                        break
            if remainder in solutions:
                result = body + "".join(solutions[remainder]) + ending
                if len(self.encode(result)) == target:
                    return result
        raise ValueError(f"cannot fit a readable block to {target} tokens with this tokenizer")

    def _make_history(self, uid: int) -> tuple[str, tuple[int, ...]]:
        history_length, _ = self.lengths_for_user(uid)
        rng = random.Random(f"{self.seed}:history:{uid}")

        def records():
            index = 0
            while True:
                index += 1
                item = rng.choice(self.catalog)
                action = rng.choice(("Viewed", "Compared", "Saved", "Purchased"))
                yield (
                    f"Record {index}: {action} P{item}, {self._title(item)}. "
                    f"Preference: {rng.choice(ATTRIBUTES)}.\n"
                )

        text = self._fit(f"User profile U{uid}.\n", records(), history_length, self.history_ending)
        return text, tuple(self.encode(text))

    def _candidates(
        self, uid: int, visit: int, item_variant: int | None = None
    ) -> tuple[str, list[int]]:
        _, item_length = self.lengths_for_user(uid)
        budget = item_length - len(self.prefix_ids)
        stream = (
            f"{self.seed}:candidates:{uid}:{visit}"
            if item_variant is None
            else f"{self.seed}:candidates:shared:{item_variant}:{visit}"
        )
        rng = random.Random(stream)
        items = rng.sample(self.catalog, self.text_config.candidate_count)
        title = f"Candidate pool for visit {visit}:\n"
        lines = [
            f"({index_letter(i)}) P{item}: {self._title(item)}.\n" for i, item in enumerate(items)
        ]
        # candidate_count is an upper bound: keep complete candidates when the
        # 64/128-token item budget cannot hold the full default pool.
        while len(items) > 1 and len(self.encode(title + "".join(lines) + self.ending)) > budget - 2:
            items.pop()
            lines.pop()
        if len(items) == 1 and len(self.encode(title + lines[0] + self.ending)) > budget - 2:
            words = self._title(items[0]).split()
            while len(words) > 1:
                words.pop()
                lines[0] = f"(A) P{items[0]}: {' '.join(words)}.\n"
                if len(self.encode(title + lines[0] + self.ending)) <= budget - 2:
                    break
            if len(self.encode(title + lines[0] + self.ending)) > budget - 2:
                lines[0] = f"(A) P{items[0]}: Catalog product.\n"
        heading = title + "".join(lines)

        def records():
            while True:
                i = rng.randrange(len(items))
                yield f"Details for ({index_letter(i)}): {rng.choice(ATTRIBUTES)}.\n"

        return self._fit(heading, records(), budget, self.ending), items

    def for_user(self, uid: int, *, visit_index: int = 0, item_variant: int | None = None) -> dict:
        if uid not in self.population.weights:
            raise KeyError(uid)
        _integer("visit_index", visit_index)
        if item_variant is not None:
            _integer("item_variant", item_variant)
        history, history_ids = self._history(uid)
        suffix, items = self._candidates(uid, visit_index, item_variant)
        suffix_ids = self.encode(suffix)
        prompt = self.prefix + history + suffix
        ids = self.encode(prompt)
        expected = self.prefix_ids + list(history_ids) + suffix_ids
        if ids != expected:
            raise ValueError(
                "tokenizer merges across section boundaries; cannot guarantee exact budgets"
            )
        h, n = self.lengths_for_user(uid)
        if len(ids) != h + n:
            raise ValueError("full prompt length does not match configured budgets")
        start = len(self.prefix_ids)
        return {
            "user_id": uid,
            "visit_index": visit_index,
            "prompt": prompt,
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "user_heat_weight": self.population.weights[uid],
            "user_tokens": h,
            "item_tokens": n,
            "instruction_tokens": start,
            "candidate_suffix_tokens": len(suffix_ids),
            "total_input_tokens": len(ids),
            "stable_prefix_tokens": start + h,
            "history_sha256": hashlib.sha256(history.encode()).hexdigest(),
            "history_token_span": [start, start + h],
            "candidate_token_span": [start + h, len(ids)],
            "candidate_item_ids": items,
            "item_content_variant": item_variant,
            "content_is_synthetic": True,
            "target_item_id": None,
        }

    def iter_generate(self, count: int):
        _integer("count", count)
        visits: Counter = Counter()
        previous: dict[int, tuple[int, float]] = {}
        for task_id, (uid, timestamp) in enumerate(
            _schedule(self.users, self.population.weights, count, self.schedule_config)
        ):
            visit = visits[uid]
            row = self.for_user(uid, visit_index=visit)
            last = previous.get(uid)
            row.update(
                task_id=task_id,
                timestamp=timestamp,
                previous_request_id=last[0] if last else None,
                revisit_interval_s=timestamp - last[1] if last else None,
                common_prefix_tokens=0,
            )
            if last:
                old_suffix, _ = self._candidates(uid, visit - 1)
                old_ids = self.encode(old_suffix)
                new_ids = row["input_ids"][row["stable_prefix_tokens"] :]
                shared = 0
                for a, b in zip(old_ids, new_ids):
                    if a != b:
                        break
                    shared += 1
                row["common_prefix_tokens"] = row["stable_prefix_tokens"] + shared
            previous[uid] = (task_id, timestamp)
            visits[uid] += 1
            yield row


def create_input_generator(
    *,
    heat_source: str = "beauty",
    heat_path: str | Path | None = None,
    industrial_heat_field: str = "pv_share",
    curve_dataset: str = "beauty",
    curve_field: str | None = None,
    num_users: int = 1000,
    text_material: str = "catalog",
    text_dataset: str = "beauty",
    text_catalog_path: str | Path | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    tokenizer: Tokenizer | str | Path = DEFAULT_TOKENIZER,
    text_config: TextConfig | None = None,
    schedule_config: ScheduleConfig | None = None,
) -> InputGenerator:
    """Load resources once and return a reusable, in-memory request generator.

    Use ``iter_generate(count)`` for a scheduled stream or ``for_user(uid,
    visit_index=...)`` for an explicit visit. Neither method writes files or
    sends requests. Each new stream restarts its schedule and visit counters.
    Paths accept strings or Path objects; tokenizer may also be preloaded.
    """
    datasets = ("beauty", "games", "books", "clothing")
    if heat_source not in (*datasets, "industrial", "curve"):
        raise ValueError(f"unsupported heat_source: {heat_source}")
    if text_material not in ("catalog", "synthetic"):
        raise ValueError(f"unsupported text_material: {text_material}")
    if text_dataset not in datasets:
        raise ValueError(f"unsupported text_dataset: {text_dataset}")
    _integer("num_users", num_users)
    schedule = schedule_config or ScheduleConfig(sampling="weighted")
    root = Path(data_root)
    if heat_source == "curve":
        population = HeatPopulation.from_curve(
            heat_path if heat_path is not None else DEFAULT_HEAT_CURVES,
            dataset=curve_dataset,
            field=curve_field
            or ("pv_share" if curve_dataset.startswith("industrial_") else "interaction_count"),
            num_users=num_users,
            seed=schedule.seed,
        )
    else:
        industrial = heat_source == "industrial"
        source = (
            Path(heat_path)
            if heat_path is not None
            else (
                root / "industrial/users_100k.csv"
                if industrial
                else root / heat_source / "timestep_map.json"
            )
        )
        population = HeatPopulation.load(
            source,
            industrial_field=industrial_heat_field if industrial else None,
            num_users=num_users,
            seed=schedule.seed,
        )
    titles = None
    if text_material == "catalog":
        catalog = (
            Path(text_catalog_path)
            if text_catalog_path is not None
            else (
                root
                / "preprocessed"
                / f"{text_dataset}_min_rating0-min_uc5-min_sc5"
                / "dataset.pkl"
            )
        )
        titles = load_titles(catalog)
    if not isinstance(tokenizer, Tokenizer):
        token_path = Path(tokenizer)
        if token_path.is_dir():
            token_path /= "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(token_path))
    return InputGenerator(
        population,
        tokenizer,
        text_config=text_config,
        schedule_config=schedule,
        titles=titles,
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heat-source",
        choices=("beauty", "games", "books", "clothing", "industrial"),
        default="beauty",
    )
    parser.add_argument(
        "--heat-path", type=Path, help="Override timestep_map.json / industrial CSV"
    )
    parser.add_argument(
        "--industrial-heat-field",
        choices=("pv_share", "pv_int", "pv_scaled_1_100"),
        default="pv_share",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=1000,
        help="Uniform user subset; 0 selects all source users",
    )
    parser.add_argument("--text-material", choices=("catalog", "synthetic"), default="catalog")
    parser.add_argument(
        "--text-dataset", choices=("beauty", "games", "books", "clothing"), default="beauty"
    )
    parser.add_argument(
        "--text-catalog-path", type=Path, help="Override dataset.pkl containing meta titles"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--user-lengths", type=int, nargs="+", default=list(DEFAULT_USER_LENGTHS))
    parser.add_argument("--user-probabilities", type=float, nargs="+")
    parser.add_argument("--item-lengths", type=int, nargs="+", default=list(DEFAULT_ITEM_LENGTHS))
    parser.add_argument("--item-probabilities", type=float, nargs="+")
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--history-cache-users", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qps", type=int, default=100)
    parser.add_argument("--arrival", choices=("poisson", "constant", "timeslot"), default="poisson")
    parser.add_argument(
        "--sampling", choices=("weighted", "uniform", "sequential"), default="weighted"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _integer("count", args.count)
        cfg = TextConfig(
            tuple(args.user_lengths),
            tuple(args.user_probabilities or [1 / len(args.user_lengths)] * len(args.user_lengths)),
            tuple(args.item_lengths),
            tuple(args.item_probabilities or [1 / len(args.item_lengths)] * len(args.item_lengths)),
            args.candidate_count,
            args.max_input_tokens,
            args.history_cache_users,
        )
        schedule = ScheduleConfig(
            seed=args.seed, qps=args.qps, arrival=args.arrival, sampling=args.sampling
        )
        gen = create_input_generator(
            heat_source=args.heat_source,
            heat_path=args.heat_path,
            industrial_heat_field=args.industrial_heat_field,
            num_users=args.num_users,
            text_material=args.text_material,
            text_dataset=args.text_dataset,
            text_catalog_path=args.text_catalog_path,
            data_root=args.data_root,
            tokenizer=args.tokenizer,
            text_config=cfg,
            schedule_config=schedule,
        )
        heat = gen.population
        catalog_path = None
        if args.text_material == "catalog":
            catalog_path = args.text_catalog_path or (
                args.data_root
                / "preprocessed"
                / f"{args.text_dataset}_min_rating0-min_uc5-min_sc5"
                / "dataset.pkl"
            )
        token_path = (
            args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary sibling so a budget failure cannot publish a partial trace.
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        counts: Counter = Counter()
        total = repeated = 0
        with partial.open("w", encoding="utf-8") as output:
            for row in gen.iter_generate(args.count):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["user_id"]] += 1
                total += row["total_input_tokens"]
                repeated += row["visit_index"] > 0
        partial.replace(args.output)
        profile_path = args.output.with_suffix(args.output.suffix + ".users.jsonl")
        with profile_path.open("w") as output:
            for uid, weight in heat.weights.items():
                h, n = gen.lengths_for_user(uid)
                output.write(
                    json.dumps(
                        {
                            "user_id": uid,
                            "weight": weight,
                            "normalized_probability": weight / heat.metadata["selected_weight_sum"],
                            "user_tokens": h,
                            "item_tokens": n,
                        }
                    )
                    + "\n"
                )
        stats = {
            "requests": args.count,
            "unique_users": len(counts),
            "repeat_requests": repeated,
            "mean_input_tokens": total / args.count if args.count else 0,
        }
        meta = {
            "schema_version": 4,
            "text_source": "rules",
            "text_material": args.text_material,
            "text_catalog_path": str(catalog_path.resolve()) if catalog_path else None,
            "heat_source": args.heat_source,
            "heat": heat.metadata,
            "text_config": asdict(cfg),
            "schedule_config": asdict(schedule),
            "tokenizer_path": str(token_path.resolve()),
            "content_is_synthetic": True,
            "length_assignment": "per_user_independent_of_heat",
            "timestamp_unit": "seconds",
            "user_profiles_path": str(profile_path.resolve()),
            "stats": stats,
        }
        args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), **stats}))


if __name__ == "__main__":
    main()
