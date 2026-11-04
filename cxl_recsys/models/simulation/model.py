"""DSA-shaped backbone forward with a reusable prefix and hidden-state output.

One packed batch and one block loop, including batch size one. Each new input
token generates Q/KV and consumes only its own sequence's causal KV. Final normalization returns every new position's state.
The reusable prefix is updated in place; suffix KV is temporary.
A failed forward invalidates the entire run; this model does not restore state.
Prompt structure, Request handling and future output processing belong to Worker.
"""

import logging

import torch

from cxl_recsys.common.timing import StageTimer
from cxl_recsys.mempool.hisparse import HiSparseCache
from cxl_recsys.models.base import BaseModel, CacheContext
from cxl_recsys.models.simulation.backbone import REFERENCE_COMMIT, DSAStub
from cxl_recsys.models.simulation.config import IndexerConfig
from cxl_recsys.models.simulation.indexer import IndexerSequence

logger = logging.getLogger(__name__)


class TestModel(BaseModel):
    __test__ = False
    name = "test"

    def __init__(
        self,
        pool,
        max_users=16,
        max_history=4096,
        layers=2,
        hot=32,
        topk=8,
        dim=64,
        *,
        indexer: IndexerConfig | None = None,
    ):
        super().__init__(
            pool, max_users, max_history, layers, hot, topk, dim, indexer=indexer
        )
        self.ops = DSAStub(dim, topk, layers, max_history, self.indexer_config)
        self.users = {}
        self.cache = HiSparseCache(
            pool, max_users * layers, max_history, hot, topk, dim
        )

    def event(self, stage, **fields):
        logger.debug("dsa_stage=%s %s", stage, fields)

    def _empty_user(self, row=None):
        """Create one user's persistent history/page/slot metadata.

        tokens/pages/slots and the base row are CPU-side Python metadata.
        index_keys holds independent BF16 history Indexer K tensors in HBM.
        MLA KV payload lives in the backing pool and HiSparse hot cache.
        The dictionary is live state referenced by both self.users and entries.
        """
        return {
            "tokens": [],
            "pages": [[] for _ in range(self.layers)],
            "slots": [[] for _ in range(self.layers)],
            "row": len(self.users) * self.layers if row is None else row,
            "index_keys": [],
        }

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
    ) -> torch.Tensor:
        """Run one packed batch; a flat single input is normalized to batch size one."""
        self._last_forward_stats = {}
        self._last_batch_stats = []
        # Compatibility is input normalization only, never another inference path.
        input_ids = list(input_ids)
        if cache is not None:
            if seq_indptr is not None or caches is not None:
                raise ValueError("Use either cache or seq_indptr/caches, not both")
            seq_indptr, caches = [0, len(input_ids)], [cache]
        elif seq_indptr is None or caches is None:
            raise ValueError("Packed forward requires both seq_indptr and caches")
        input_ids, positions, seq_indptr, caches = (
            list(input_ids),
            list(positions),
            list(seq_indptr),
            list(caches),
        )
        if (
            not caches
            or len(seq_indptr) != len(caches) + 1
            or any(type(offset) is not int for offset in seq_indptr)
            or seq_indptr[0] != 0
            or seq_indptr[-1] != len(input_ids)
            or len(positions) != len(input_ids)
            or any(a >= b for a, b in zip(seq_indptr, seq_indptr[1:]))
            or any(not isinstance(cache, CacheContext) for cache in caches)
        ):
            raise ValueError("Invalid packed sequence boundaries or cache contexts")
        if any(
            not isinstance(cache.cache_key, str) or not cache.cache_key
            for cache in caches
        ):
            raise ValueError("A nonempty cache key is required")
        if len({cache.cache_key for cache in caches}) != len(caches):
            raise ValueError("Batch cache keys must be unique")
        new_users = sum(cache.cache_key not in self.users for cache in caches)
        if len(self.users) + new_users > self.max_users:
            raise ValueError("User capacity exhausted")

        entries, packed_ids, packed_positions, query_indptr = [], [], [], [0]
        next_row = len(self.users) * self.layers
        # Validate every member before allocating backing pages or modifying a user.
        for i, cache in enumerate(caches):
            start, stop = seq_indptr[i : i + 2]
            ids, pos = self._validate_forward(
                input_ids[start:stop], positions[start:stop], cache
            )
            state = self.users.get(cache.cache_key)
            if state is None:
                state = self._empty_user(row=next_row)
                next_row += self.layers
            # Capture the history length before this call mutates the live state.
            history_length = len(state["tokens"])
            # Total cacheable history after this call, excluding candidate tokens.
            total_length = cache.cacheable_prefix_length
            if total_length < history_length or ids[:history_length] != state["tokens"]:
                raise ValueError(
                    "Cached prefix is append-only; cached prefix cannot change"
                )
            # Additional backing pages per layer, not a token count or layer sum.
            new_pages_num = (
                total_length + self.pool.page_size - 1
            ) // self.pool.page_size - len(state["pages"][0])
            entries.append(
                {
                    "cache": cache,
                    "state": state,
                    "history_length": history_length,
                    "total_length": total_length,
                    "new_pages_num": new_pages_num,
                    "stats": {
                        "hits": 0,
                        "misses": 0,
                        "local_tokens": 0,
                        "queries": 0,
                        "old_references": 0,
                        "deduplicated_references": 0,
                        "h2d_transfers": 0,
                        "cxl_read_kernel_launches": 0,
                        "planner_calls": 0,
                        "new_pages": new_pages_num * self.layers,
                        "new_tokens": (total_length - history_length) * self.layers,
                    },
                }
            )
            packed_ids.extend(ids[history_length:])
            packed_positions.extend(pos[history_length:])
            query_indptr.append(len(packed_ids))
            # Entries keep metadata only; drop temporary per-request token copies.
            del ids, pos

        hidden = self.ops.embed(packed_ids)
        residual = None
        current_positions = torch.tensor(
            packed_positions, device="cuda", dtype=torch.int64
        )
        self.event(
            "embed",
            batch_size=len(caches),
            input_tokens=len(packed_ids),
            query_indptr=query_indptr,
        )
        # State is live, not transactional. Discard the run if a later stage fails.
        allocated = self.pool.allocate(
            sum(entry["stats"]["new_pages"] for entry in entries)
        )
        cursor = 0
        for i, entry in enumerate(entries):
            state, new_pages_num = entry["state"], entry["new_pages_num"]
            self.users[entry["cache"].cache_key] = state
            # Append only new history from the original packed input. No entry
            # needs another full history or candidate token list.
            history_start = seq_indptr[i] + entry["history_length"]
            history_end = seq_indptr[i] + entry["total_length"]
            state["tokens"].extend(input_ids[history_start:history_end])
            if not state["index_keys"]:
                state["index_keys"] = [
                    torch.empty(
                        (self.max_history, self.indexer_config.head_dim),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    for _ in range(self.layers)
                ]
            for layer in range(self.layers):
                state["pages"][layer].extend(allocated[cursor : cursor + new_pages_num])
                cursor += new_pages_num

        if timing is not None:
            timing.mark("prepare_ms")
        # The only transformer-layer loop covers the entire packed batch.
        for layer in range(self.layers):
            hidden, residual = self._forward_layer(
                entries,
                query_indptr,
                hidden,
                residual,
                current_positions,
                layer,
                timing=timing,
            )
        if timing is not None:
            timing.mark("layers_ms", synchronize=False)
            timing.values["layer_overhead_ms"] = timing.values["layers_ms"] - sum(
                item["total_ms"] for item in timing.layers
            )
        hidden_states = self.ops.final_norm(hidden, residual)
        self.event("final_norm", batch_size=len(entries), input_tokens=len(packed_ids))
        diagnostics = []
        for i, entry in enumerate(entries):
            count = query_indptr[i + 1] - query_indptr[i]
            diagnostics.append(
                {
                    "kv_length": entry["total_length"],
                    "mode": "extend"
                    if entry["total_length"] > entry["history_length"]
                    else "reuse",
                    "input_tokens": count,
                    "cache": entry["stats"],
                    "model": self.name,
                    "simulated": True,
                    "flow": "deepseek_v3_2_exp",
                    "reference_commit": REFERENCE_COMMIT,
                    "indexer": {
                        "mode": "dsa_random_weights",
                        "weight_seed": self.indexer_config.seed,
                        "index_key_cache": True,
                        "key_cache_device": "cuda",
                        "key_cache_dtype": "bfloat16",
                        "n_heads": self.indexer_config.n_heads,
                        "head_dim": self.indexer_config.head_dim,
                        "q_lora_rank": self.indexer_config.q_lora_rank,
                        "rope_dim": self.indexer_config.rope_dim,
                        "query_tile": self.indexer_config.query_tile,
                        "history_key_bytes": entry["total_length"]
                        * self.layers
                        * self.indexer_config.head_dim
                        * 2,
                        "allocated_key_bytes": self.max_history
                        * self.layers
                        * self.indexer_config.head_dim
                        * 2,
                    },
                    "batch": {
                        "size": len(entries),
                        "position": i,
                        "query_start": query_indptr[i],
                        "query_end": query_indptr[i + 1],
                    },
                    "kv_layout": {
                        "latent_dim": self.ops.latent_dim,
                        "rope_dim": self.ops.rope_dim,
                    },
                    "pool": self.pool.stats(),
                }
            )
        self.event(
            "forward_finished", batch_size=len(entries), new_pages=len(allocated)
        )
        self._last_batch_stats = diagnostics
        # The legacy diagnostics accessor is only a view of the same batch result.
        self._last_forward_stats = diagnostics[0] if len(diagnostics) == 1 else {}
        if timing is not None:
            timing.mark("finalize_ms")
        return hidden_states

    def _validate_forward(self, input_ids, positions, cache):
        """Validate numeric model inputs before allocating pages or touching KV."""
        input_ids = list(input_ids)
        positions = list(positions)
        if not input_ids or any(
            type(t) is not int or not -(2**31) <= t < 2**31 for t in input_ids
        ):
            raise ValueError("input_ids must be a nonempty sequence of int32 token IDs")
        if any(type(p) is not int for p in positions) or positions != list(
            range(len(input_ids))
        ):
            raise ValueError("positions must describe the complete sequence 0..N-1")
        if (
            not isinstance(cache, CacheContext)
            or not isinstance(cache.cache_key, str)
            or not cache.cache_key
        ):
            raise ValueError("A nonempty cache key is required")
        prefix_length = cache.cacheable_prefix_length
        if type(prefix_length) is not int or not 0 <= prefix_length <= min(
            self.max_history, len(input_ids)
        ):
            raise ValueError("Cacheable prefix capacity exceeded or invalid boundary")
        # This adapter only returns newly computed states, never cached states.
        state = self.users.get(cache.cache_key)
        if state is not None and len(state["tokens"]) == len(input_ids):
            raise ValueError("Forward requires at least one uncached input token")
        return input_ids, positions

    def _forward_layer(
        self,
        entries,
        query_indptr,
        hidden,
        residual,
        positions,
        layer,
        timing=None,
    ):
        """Batched projection -> history write -> Indexer -> HiSparse -> attention."""
        layer_timing = StageTimer(timing.synchronize) if timing is not None else None
        self.event(
            "layer_started", layer=layer, batch_size=len(entries), queries=len(hidden)
        )
        # 1. Normalize all current tokens across the batch on GPU. Cached prefix
        # tokens are omitted; each sequence keeps its own absolute positions.
        block_residual = hidden if residual is None else (hidden + residual).bfloat16()
        normalized = self.ops.norm(block_residual)
        # 2. Generate Q and compressed MLA KV for every packed current token once.
        # q_nope/q_pe feed attention; KV stays in HBM. The normalized low-rank
        # query qr is shared with the independent Indexer Q projection below.
        projected = self.ops.project(normalized, positions, layer)
        self.event("q_kv_projected", layer=layer, batch_size=len(entries))

        # 3. Map new history [history_length, total_length) through each layer page table.
        # query_indptr addresses packed current tokens, while slots address physical
        # backing-pool tokens. Candidate suffix KV must never enter the history pool.
        # These per-user loops only prepare metadata; they do not run model blocks.
        write_slots, write_rows = [], []
        for i, entry in enumerate(entries):
            state = entry["state"]
            history_length = entry["history_length"]
            total_length = entry["total_length"]
            pages = state["pages"][layer]
            slots = [
                self.pool.token_slot(
                    pages[pos // self.pool.page_size], pos % self.pool.page_size
                )
                for pos in range(history_length, total_length)
            ]
            state["slots"][layer].extend(slots)
            write_slots.extend(slots)
            write_rows.extend(
                range(query_indptr[i], query_indptr[i] + total_length - history_length)
            )
            self.cache.bind(state["row"] + layer, state["slots"][layer])
        if layer_timing is not None:
            # Includes normalization, Q/KV projection and destination-slot mapping.
            layer_timing.mark("before_kv_write_ms")
        # 4. Write the entire batch's new history with one pool call for this layer.
        # CXL uses the registered devdax mapping directly from the CUDA kernel;
        # RDMA registers HBM directly; only local DRAM uses host staging.
        if write_slots:
            history_kv = projected.kv[write_rows].contiguous()
            if self.pool.backend == "dram":
                history_kv = history_kv.cpu()
            self.pool.write_many(write_slots, history_kv)
        self.event(
            "mla_cache_written",
            layer=layer,
            batch_size=len(entries),
            history_tokens=len(write_slots),
            temporary_tokens=len(hidden) - len(write_slots),
        )
        if layer_timing is not None:
            # Includes packing and transport; only DRAM includes host staging.
            layer_timing.mark("kv_write_ms")

        # 5. Absorb the KV expansion projection into Q (identity in this stub).
        projected.q_nope = self.ops.absorb_query(projected.q_nope)
        self.event("query_absorbed", layer=layer, batch_size=len(entries))
        if layer_timing is not None:
            layer_timing.mark("query_absorb_ms")
        # 6. Project Indexer Q/K/head weights, append only history K in HBM,
        # and compute weighted ReLU logits plus causal Top-K for the whole layer.
        # Cached Indexer K is independent of MLA KV; no full-history CXL read
        # precedes selection. Candidate keys are temporary, never persistent.
        sequences = [
            IndexerSequence(
                history_keys=entry["state"]["index_keys"][layer],
                cached_length=entry["history_length"],
                history_length=entry["total_length"],
                query_start=query_indptr[i],
                query_end=query_indptr[i + 1],
            )
            for i, entry in enumerate(entries)
        ]
        selected = self.ops.select_batch(
            positions, normalized, projected.qr, layer, sequences
        )
        self.event(
            "index_cache_written",
            layer=layer,
            history_tokens=sum(
                entry["total_length"] - entry["history_length"] for entry in entries
            ),
            temporary_tokens=len(hidden) - len(write_slots),
            device="cuda",
            dtype="bfloat16",
        )
        self.event(
            "index_topk",
            layer=layer,
            batch_size=len(entries),
            queries=len(hidden),
            topk=selected.shape[1],
            mode="dsa_random_weights",
        )
        if layer_timing is not None:
            # Includes Indexer projection/rotation, HBM key append, scoring and
            # selection. Shared qr projection is charged to before_kv_write_ms.
            layer_timing.mark("indexer_ms")

        # 7. Hand every user's Top-K slice to one multi-row HiSparse call.
        # Each row is user_base_row + layer. HiSparse deduplicates old KV, serves
        # HBM hits, and merges misses across users into one backing-pool transfer.
        # Newly generated history and candidate KV are consumed directly from HBM.
        # HiSparse names these boundaries differently: history_length is the
        # updated history size, while cached_length is the pre-forward size.
        fetches = []
        for i, entry in enumerate(entries):
            start, stop = query_indptr[i : i + 2]
            fetches.append(
                {
                    "row": entry["state"]["row"] + layer,
                    "indices": selected[start:stop],
                    "history_length": entry["total_length"],
                    "cached_length": entry["history_length"],
                    "current_kv": projected.kv[start:stop],
                }
            )
        batch, counters = self.cache.fetch_layer(fetches)
        if not isinstance(counters, list):
            raise RuntimeError("TestModel requires synchronous cache counters")
        for i, (entry, counts) in enumerate(zip(entries, counters)):
            for name, count in counts.items():
                entry["stats"][name] += count
            entry["stats"]["queries"] += query_indptr[i + 1] - query_indptr[i]
        if layer_timing is not None:
            # Whole HiSparse path: planning, hits, misses, transfers and mapping.
            # A task with no historical misses still has planning overhead here.
            layer_timing.mark("kv_read_ms")

        # 8. One packed consumption call. Each query has already been mapped to only
        # its user's compact KV rows; padding remains -1, never another user's row.
        attention = self.ops.consume(
            projected.q_nope, projected.q_pe, batch.values, batch.indices
        )
        self.event(
            "attention_consumed",
            layer=layer,
            batch_size=len(entries),
            queries=len(hidden),
        )
        if layer_timing is not None:
            layer_timing.mark("attention_ms")
        # 9. Run the attention residual/FFN stub over the packed batch. Both
        # tensors feed the next layer; final normalization happens after all layers.
        hidden, residual = self.ops.block_output(attention, block_residual)
        self.event("ffn_finished", layer=layer, batch_size=len(entries))
        self.event("layer_finished", layer=layer, batch_size=len(entries))
        if layer_timing is not None and timing is not None:
            layer_timing.mark("ffn_ms")
            timing.layers.append({"layer": layer, **layer_timing.report()})
        return hidden, residual

    def synchronize(self) -> None:
        """This prototype has one CUDA device per Worker."""
        torch.cuda.synchronize()

    def snapshot(self):
        return {
            uid: {"kv_length": len(u["tokens"]), "backend": self.pool.backend}
            for uid, u in self.users.items()
        }
