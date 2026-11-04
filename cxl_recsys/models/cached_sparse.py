"""Shared packed prefix lifecycle and candidate scoring for sparse backbones.

Attention and persistent summary layouts remain model-specific. Only immutable
history enters the backing pool; candidates are temporary in every adapter.
"""

from pathlib import Path
from string import ascii_uppercase

import torch
import torch.nn.functional as F

from cxl_recsys.models import tensor_ops as ops

from ..mempool.hisparse import HiSparseCache
from .base import BaseModel, CacheContext


class CachedSparseModel(BaseModel):
    flow: str
    source_revision: str
    candidate_head_key = "lm_head.weight"
    answer_boundary = "\nAnswer:"
    position_limit: int | None = None

    def __init__(
        self,
        pool,
        max_users,
        max_history,
        layers,
        hot,
        topk,
        dim,
        *,
        config,
        model_path,
        cache_backend="reference",
    ):
        super().__init__(pool, max_users, max_history, layers, hot, topk, dim)
        self.config = config
        self.model_path = str(Path(model_path).resolve())
        self.heads = config.num_key_value_heads
        self.groups = config.num_attention_heads // self.heads
        self.users = {}
        self.failed = False
        self.cache = HiSparseCache(
            pool,
            max_users * layers * self.heads,
            max_history,
            hot,
            topk,
            dim,
            device_resident=cache_backend == "device",
        )
        self.weights = {}
        self._candidate_head = None
        self._candidate_tokenizer = None
        self._candidate_token_ids = []

    def _empty_user(self, row):
        raise NotImplementedError

    def _layer(self, hidden, positions, entries, indptr, layer, timing):
        raise NotImplementedError

    def _normalize_output(self, hidden):
        return ops.rms_norm(
            hidden, self.weights["model.norm.weight"], self.config.rms_norm_eps
        )

    def _load_candidate_head(self):
        """Load only A-Z output rows, lazily for recommendation requests."""
        from safetensors import safe_open
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(self.model_path) / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        prefix = tokenizer.encode(self.answer_boundary, add_special_tokens=False).ids
        token_ids = []
        for option in ascii_uppercase:
            encoded = tokenizer.encode(
                self.answer_boundary + option, add_special_tokens=False
            ).ids
            if (
                len(encoded) != len(prefix) + 1
                or encoded[:-1] != prefix
                or not 0 <= encoded[-1] < self.config.vocab_size
            ):
                raise ValueError(
                    "sparse model answer options must append exactly one token"
                )
            token_ids.append(encoded[-1])
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("sparse model answer options must have distinct token IDs")
        with safe_open(
            Path(self.model_path) / "model.safetensors", framework="pt", device="cpu"
        ) as source:
            head = source.get_slice(self.candidate_head_key)
            if (
                tuple(head.get_shape())
                != (self.config.vocab_size, self.config.hidden_size)
                or head.get_dtype() != "BF16"
            ):
                raise ValueError("Invalid sparse model lm_head shape/dtype")
            rows = torch.cat([head[token : token + 1] for token in token_ids])
        self._candidate_head = rows.to(
            device=self.weights["model.norm.weight"].device, dtype=torch.float32
        )
        self._candidate_tokenizer = tokenizer
        self._candidate_token_ids = token_ids

    @torch.inference_mode()
    def select_candidate_options(self, hidden_states, output_ends, suffixes, counts):
        """Score immediate A-Z completions; return CPU (option, index) pairs.

        output_ends are exclusive boundaries in this forward's packed output,
        not in complete prompts. No selected answer is appended to history KV.
        """
        if (
            not counts
            or len(output_ends) != len(counts)
            or len(suffixes) != len(counts)
        ):
            raise ValueError("Invalid candidate selection batch")
        if any(type(n) is not int or not 1 <= n <= 26 for n in counts):
            raise ValueError(
                "sparse model candidate selection supports 1..26 options (A-Z)"
            )
        if any(
            type(end) is not int or not 1 <= end <= len(hidden_states)
            for end in output_ends
        ):
            raise ValueError("Invalid candidate output boundary")
        if self._candidate_head is None:
            self._load_candidate_head()
        tokenizer = self._candidate_tokenizer
        assert tokenizer is not None and self._candidate_head is not None
        for ids, count in zip(suffixes, counts):
            text = tokenizer.decode(ids, skip_special_tokens=False)
            if not text.endswith(self.answer_boundary):
                raise ValueError(
                    f"Candidate prompt must end with {self.answer_boundary!r}"
                )
            for option, token in zip(
                ascii_uppercase[:count], self._candidate_token_ids
            ):
                if tokenizer.encode(text + option, add_special_tokens=False).ids != [
                    *ids,
                    token,
                ]:
                    raise ValueError(
                        "sparse model candidate answer token boundary mismatch"
                    )
        last = hidden_states[
            torch.tensor(output_ends, device=hidden_states.device) - 1
        ].float()
        logits = F.linear(last, self._candidate_head[: max(counts)])
        limits = torch.tensor(counts, device=logits.device)
        logits.masked_fill_(
            torch.arange(logits.shape[1], device=logits.device)[None]
            >= limits[:, None],
            -torch.inf,
        )
        # tolist waits for scoring/argmax and the device-to-host result transfer.
        selected = logits.argmax(-1).tolist()
        return [(ascii_uppercase[index], index) for index in selected]

    def _prepare(self, input_ids, positions, seq_indptr, caches):
        if (
            not caches
            or len(seq_indptr) != len(caches) + 1
            or any(type(n) is not int for n in seq_indptr)
            or seq_indptr[0] != 0
            or seq_indptr[-1] != len(input_ids)
            or len(input_ids) != len(positions)
            or any(a >= b for a, b in zip(seq_indptr, seq_indptr[1:]))
        ):
            raise ValueError("Invalid packed sparse model sequence boundaries")
        if any(
            not isinstance(c, CacheContext)
            or not isinstance(c.cache_key, str)
            or not c.cache_key
            for c in caches
        ):
            raise ValueError("sparse model requires nonempty cache keys")
        if len({c.cache_key for c in caches}) != len(caches):
            raise ValueError("Batch cache keys must be unique")
        if (
            len(self.users) + sum(c.cache_key not in self.users for c in caches)
            > self.max_users
        ):
            raise ValueError("User capacity exhausted")
        entries, new_ids, new_positions, query_indptr = [], [], [], [0]
        next_row = len(self.users) * self.layers * self.heads
        for i, context in enumerate(caches):
            start, stop = seq_indptr[i : i + 2]
            ids, pos = input_ids[start:stop], positions[start:stop]
            if any(
                type(t) is not int or not 0 <= t < self.config.vocab_size for t in ids
            ):
                raise ValueError("sparse model token ID outside checkpoint vocabulary")
            if any(type(p) is not int for p in pos) or pos != list(range(len(ids))):
                raise ValueError("positions must describe the complete sequence 0..N-1")
            if self.position_limit is not None and len(ids) > self.position_limit:
                raise ValueError("Input exceeds checkpoint position limit")
            prefix = context.cacheable_prefix_length
            if type(prefix) is not int or not 0 <= prefix <= min(
                self.max_history, len(ids)
            ):
                raise ValueError("Invalid cacheable prefix length")
            state = self.users.get(context.cache_key)
            old = len(state["tokens"]) if state is not None else 0
            if old > prefix or (state is not None and ids[:old] != state["tokens"]):
                raise ValueError(
                    "Cached prefix is append-only; cached prefix cannot change"
                )
            if len(ids) == old:
                raise ValueError("Forward requires at least one uncached input token")
            if state is None:
                state = self._empty_user(next_row)
                next_row += self.layers * self.heads
            page_count = (prefix + self.pool.page_size - 1) // self.pool.page_size
            added_pages = page_count - len(state["pages"][0][0])
            entries.append(
                {
                    "context": context,
                    "state": state,
                    "old": old,
                    "prefix": prefix,
                    "total": len(ids),
                    "ids": ids,
                    "added_pages": added_pages,
                    "stats": {
                        name: 0
                        for name in (
                            "hits",
                            "misses",
                            "old_references",
                            "deduplicated_references",
                            "local_tokens",
                            "planner_calls",
                            "h2d_transfers",
                            "cxl_read_kernel_launches",
                            "queries",
                        )
                    },
                }
            )
            new_ids.extend(ids[old:])
            new_positions.extend(pos[old:])
            query_indptr.append(len(new_ids))
        needed = sum(e["added_pages"] for e in entries) * self.layers * self.heads
        pages = iter(self.pool.allocate(needed))
        for entry in entries:
            state = entry["state"]
            self.users[entry["context"].cache_key] = state
            state["tokens"] = entry["ids"][: entry["prefix"]]
            for layer in range(self.layers):
                for head in range(self.heads):
                    state["pages"][layer][head].extend(
                        next(pages) for _ in range(entry["added_pages"])
                    )
            entry["stats"]["new_pages"] = (
                entry["added_pages"] * self.layers * self.heads
            )
            entry["stats"]["new_tokens"] = (
                entry["prefix"] - entry["old"]
            ) * self.layers
        return entries, new_ids, new_positions, query_indptr

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        positions,
        *,
        cache=None,
        seq_indptr=None,
        caches=None,
        timing=None,
    ):
        self._last_forward_stats = {}
        self._last_batch_stats = []
        self._pending_cache_stats = []
        if self.failed:
            raise RuntimeError(
                "sparse model instance failed; discard it instead of retrying"
            )
        input_ids, positions = list(input_ids), list(positions)
        if cache is not None:
            if seq_indptr is not None or caches is not None:
                raise ValueError("Use cache or seq_indptr/caches, not both")
            seq_indptr, caches = [0, len(input_ids)], [cache]
        if seq_indptr is None or caches is None:
            raise ValueError("Packed forward requires seq_indptr and caches")
        # A failure may follow cache/page mutation. Never retry this instance.
        try:
            entries, ids, pos, indptr = self._prepare(
                input_ids, positions, list(seq_indptr), list(caches)
            )
            ids_gpu = torch.tensor(ids, device="cuda", dtype=torch.long)
            positions_gpu = torch.tensor(pos, device="cuda", dtype=torch.long)
            hidden = F.embedding(ids_gpu, self.weights["model.embed_tokens.weight"])
            if timing is not None:
                timing.mark("prepare_ms")
            for layer in range(self.layers):
                hidden = self._layer(
                    hidden, positions_gpu, entries, indptr, layer, timing
                )
            if timing is not None:
                timing.mark("layers_ms", synchronize=False)
                timing.values["layer_overhead_ms"] = timing.values["layers_ms"] - sum(
                    item["total_ms"] for item in timing.layers
                )
            hidden = self._normalize_output(hidden)
            if self._pending_cache_stats:
                rows = torch.cat(
                    [item.statistics for _, item in self._pending_cache_stats]
                ).tolist()
                cursor = 0
                for owners, item in self._pending_cache_stats:
                    count = len(item.statistics)
                    for i, counter in enumerate(
                        item.materialize(rows[cursor : cursor + count])
                    ):
                        stats = owners[i // self.heads]["stats"]
                        for name, value in counter.items():
                            stats[name] += value
                    cursor += count
                self._pending_cache_stats = []
            self.pool.flush_read_stats()
            self._last_batch_stats = [
                {
                    "model": self.name,
                    "simulated": False,
                    "flow": self.flow,
                    "kv_length": entry["prefix"],
                    "mode": "extend" if entry["prefix"] > entry["old"] else "reuse",
                    "input_tokens": indptr[i + 1] - indptr[i],
                    "cache": entry["stats"],
                    "pool": self.pool.stats(),
                    "kv_layout": {
                        "heads": self.heads,
                        "head_dim": self.config.head_dim,
                        "record_dim": self.dim,
                        "record_bytes": self.dim * 2,
                    },
                    "batch": {
                        "size": len(entries),
                        "position": i,
                        "query_start": indptr[i],
                        "query_end": indptr[i + 1],
                    },
                    "reference": {
                        "source": self.source_revision,
                        "checkpoint_path": self.model_path,
                    },
                }
                for i, entry in enumerate(entries)
            ]
            if len(entries) == 1:
                self._last_forward_stats = self._last_batch_stats[0]
            if timing is not None:
                timing.mark("finalize_ms")
            return hidden
        except Exception:
            self.failed = True
            self._last_forward_stats = {}
            self._last_batch_stats = []
            raise

    def synchronize(self):
        torch.cuda.synchronize()

    def snapshot(self):
        return {
            key: {"kv_length": len(state["tokens"]), "backend": self.pool.backend}
            for key, state in self.users.items()
        }

    def close(self):
        self.synchronize()
        self.users.clear()
        self.weights.clear()
        super().close()
