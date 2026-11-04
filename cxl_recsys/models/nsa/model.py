"""FLA mean-pooling NSA with HiSparse-backed persistent history and GR output."""

from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from cxl_recsys.common.timing import StageTimer
from cxl_recsys.models import tensor_ops as common
from cxl_recsys.models.cached_sparse import CachedSparseModel
from cxl_recsys.models.nsa import ops
from cxl_recsys.models.nsa.config import NSA_REVISION, NSAConfig


class NSAModel(CachedSparseModel):
    name = "nsa_1b"
    flow = "nsa_mean_pool"
    source_revision = NSA_REVISION
    answer_boundary = "\nAnswer:\n"

    def __init__(
        self,
        pool,
        max_users=16,
        max_history=4096,
        layers=16,
        hot=256,
        topk=256,
        dim=128,
        *,
        model_path,
        config=None,
        query_tile=32,
        kernel_backend="fla",
        cuda_graphs=False,
        graph_cache_size=4,
        cache_backend="auto",
        selected_backend="triton",
        cross_query_dedup=True,
        rdma_registration="persistent",
        selected_mode="overlap",
        selected_order="strict",
        selected_ready_window=1,
        selected_read_order="physical",
        selected_read_group_size=16,
        selected_max_queries=128,
        selected_query_group=1,
        selected_workspace_mib=512,
        indexer=None,
    ):
        from cxl_recsys.models.nsa.planning.read_order import READ_ORDERS

        if selected_ready_window not in (1, 2, 4):
            raise ValueError("NSA selected_ready_window must be 1, 2 or 4")
        if selected_ready_window > 1 and selected_backend != "tilelang_tma":
            raise ValueError("NSA ready windows require the TMA consumer")
        if selected_read_order not in READ_ORDERS or selected_read_group_size < 1:
            raise ValueError("Invalid NSA read priority configuration")
        if selected_read_order != "physical" and selected_backend not in (
            "tilelang_gr",
            "tilelang_tma",
            "triton_gr",
        ):
            raise ValueError("Read priority requires tilelang_gr")
        if selected_order not in ("strict", "resident_first", "block_ready"):
            raise ValueError("Unknown NSA consumption order")
        if selected_order != "strict" and selected_backend not in (
            "tilelang_gr",
            "tilelang_tma",
            "triton_gr",
        ):
            raise ValueError("Ready-order consumption requires tilelang_gr")
        if (
            selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr")
            and selected_query_group != 1
        ):
            # Fixed-selection agreement does not guarantee that later Top-K
            # decisions stay within the complete-model numerical tolerance.
            raise ValueError(
                "NSA GR shared query groups are operator-only experiments: "
                "full-model numerical validation failed; use selected_query_group=1"
            )
        cfg = config or NSAConfig.load(model_path)
        if (layers, dim, topk) != (
            cfg.num_hidden_layers,
            cfg.record_dim,
            cfg.selected_tokens,
        ):
            raise ValueError("NSA model/cache geometry does not match checkpoint")
        if query_tile <= 0:
            raise ValueError("NSA query_tile must be positive")
        if kernel_backend not in ("fla", "torch"):
            raise ValueError("NSA kernel_backend must be fla or torch")
        if graph_cache_size <= 0:
            raise ValueError("NSA graph_cache_size must be positive")
        if cache_backend not in ("auto", "reference", "device"):
            raise ValueError("NSA cache_backend must be auto, reference or device")
        if cache_backend == "auto":
            cache_backend = "device" if pool.backend in ("dram", "cxl") else "reference"
        if pool.pages < cfg.required_pages(max_users, max_history, pool.page_size):
            raise ValueError("NSA backing pool capacity is too small")
        super().__init__(
            pool,
            max_users,
            max_history,
            layers,
            hot,
            topk,
            dim,
            config=cfg,
            model_path=model_path,
            cache_backend=cache_backend,
        )
        self.query_tile = query_tile
        self.kernel_backend = kernel_backend
        self.cache_backend = cache_backend
        if selected_backend not in (
            "triton",
            "triton_gr",
            "tilelang",
            "tilelang_indexed",
            "tilelang_gpu",
            "tilelang_gr",
            "tilelang_tma",
        ):
            raise ValueError("Unsupported NSA selected attention backend")
        if not cross_query_dedup and selected_backend not in (
            "triton_gr",
            "tilelang_tma",
        ):
            raise ValueError("Query-private KV requires triton_gr or tilelang_tma")
        if selected_max_queries < 1:
            raise ValueError("NSA selected_max_queries must be positive")
        if selected_backend == "tilelang_tma" and (
            self.groups != 16 or cfg.block_size != 16
        ):
            raise ValueError("TMA selected attention requires G16 and BS16")
        self.selected_backend = selected_backend
        self.cross_query_dedup = cross_query_dedup
        if selected_query_group not in (1, 2, 4, 8) or selected_workspace_mib < 1:
            raise ValueError("Invalid NSA GR query group or workspace budget")
        self.selected_max_queries = selected_max_queries
        self.selected_calls = self.selected_fallbacks = 0
        self.selected_fallback_reasons = {}
        self.selected_request_info = {}
        self._gr_allowed = False
        self.selected_transport = None
        if selected_backend.startswith("tilelang"):
            import importlib.util

            if importlib.util.find_spec("tilelang") is None:
                raise RuntimeError(
                    "TileLang NSA requires the isolated tools/nsa-tilelang uv environment; source scripts/nsa_tilelang_environment.sh"
                )
            if torch.cuda.get_device_capability() != (9, 0):
                raise ValueError("TileLang NSA currently requires H100 / SM90")
        if selected_backend == "tilelang":
            from cxl_recsys.models.nsa.backends.legacy_transport import (
                SelectedTransport,
            )

            self.selected_transport = SelectedTransport(self.cache, mode=selected_mode)
        elif selected_backend == "tilelang_gpu":
            from cxl_recsys.models.nsa.backends.gpu_transport import (
                GPUSelectedTransport,
            )

            self.selected_transport = GPUSelectedTransport(
                self.cache,
                mode=selected_mode,
                cuda_graphs=cuda_graphs,
                capacity=graph_cache_size,
            )
        elif selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr"):
            from cxl_recsys.models.nsa.transport import GRSelectedTransport

            self.selected_transport = GRSelectedTransport(
                self.cache,
                mode=selected_mode,
                cuda_graphs=cuda_graphs,
                capacity=graph_cache_size,
                consumer_backend=(
                    "tma"
                    if selected_backend == "tilelang_tma"
                    else "native"
                    if selected_backend == "triton_gr"
                    else "legacy"
                ),
                cross_query_dedup=cross_query_dedup,
                rdma_registration=rdma_registration,
                query_group=selected_query_group,
                consume_order=selected_order,
                ready_window=selected_ready_window,
                read_order=selected_read_order,
                read_group_size=selected_read_group_size,
                workspace_bytes=selected_workspace_mib * 1024**2,
            )
        from cxl_recsys.models.cuda_graph import CUDAGraphCache

        self.graphs = CUDAGraphCache(graph_cache_size) if cuda_graphs else None
        # Projection/finish graphs bind each layer's immutable weights. Bound
        # capacity permits the same number of query shapes per layer/phase.
        self.compute_graphs = (
            CUDAGraphCache(graph_cache_size * layers * 3) if cuda_graphs else None
        )
        self.gr_compute = self.gr_local = None
        self.gr_finish_down_calls = 0
        self._gr_bucket_execution = False
        if selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr"):
            from cxl_recsys.models.nsa.dispatch import NSABucketExecutor

            self.gr_compute = NSABucketExecutor(
                graph_cache_size * layers * 3,
                cuda_graphs=cuda_graphs,
                direct_replay=True,
            )
            self.gr_local = NSABucketExecutor(
                graph_cache_size, cuda_graphs=cuda_graphs, direct_replay=True
            )
        if cuda_graphs and self.cache.device_resident:
            self.cache.plan_graphs = CUDAGraphCache(graph_cache_size)
        self.position_limit = cfg.max_position_embeddings
        self.weights = self._load_weights(Path(model_path))
        exponent = (
            torch.arange(0, cfg.head_dim, 2, device="cuda").float() / cfg.head_dim
        )
        self.inv_freq = 1.0 / cfg.rope_theta**exponent

    def _local_branches(
        self,
        query,
        compressed,
        window,
        positions,
        total,
        window_start,
        *,
        return_blocks=False,
        borrow=False,
    ):
        cfg = self.config
        options = dict(
            total=total,
            window_start=window_start,
            block_size=cfg.block_size,
            block_counts=cfg.block_counts,
            window_size=cfg.window_size,
            tile=self.query_tile,
            backend=self.kernel_backend,
            return_blocks=return_blocks,
        )

        def execute(*inputs):
            if self._gr_bucket_execution and self.kernel_backend == "torch":
                # The torch oracle's current-block scatter requires a valid
                # index even for scratch padding. Discard its dummy selection.
                q, summaries, records, pos = inputs
                output, local, selected = ops.local_branches(
                    q, summaries, records, pos.clamp_min(0), **options
                )
                return output, local, selected.masked_fill(pos[:, None, None] < 0, -1)
            return ops.local_branches(*inputs, **options)

        inputs = (query, compressed, window, positions)
        with torch.cuda.nvtx.range("nsa.local_branches"):
            if self._gr_bucket_execution:
                # projection_prepare has already created a full-capacity window
                # and summary. Future padded summaries cannot pass the causal
                # complete-block mask for any real query.
                options["total"] = window_start + len(window)
                return self.gr_local.run(
                    tuple(options.items()),
                    execute,
                    inputs,
                    active_q=len(query),
                    pad_inputs=(0, 3),
                    fill_values={3: -1},
                    clone_outputs=not borrow,
                )
            if self.graphs is None:
                return execute(*inputs)
            return self.graphs.run(
                tuple(options.items()), execute, inputs, clone_outputs=not borrow
            )

    def _execution_stats(self):
        return {
            "kernel_backend": self.kernel_backend,
            "cache_backend": self.cache_backend,
            "selected_backend": self.selected_backend,
            "selected_cross_query_dedup": self.cross_query_dedup,
            "selected_tilelang_calls": self.selected_calls,
            # Legacy tilelang_calls counts adapter invocations. GR may use the
            # bounded native consumer; its precise consumer is reported below.
            "selected_transport_calls": self.selected_calls,
            "selected_gr_calls": self.selected_calls
            if self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr")
            else 0,
            "selected_tma_calls": self.selected_calls
            if self.selected_backend == "tilelang_tma"
            else 0,
            "selected_native_fallbacks": self.selected_fallbacks,
            "selected_fallback_reasons": dict(self.selected_fallback_reasons),
            "selected_request": dict(self.selected_request_info),
            "selected_transport": self.selected_transport.last_stats
            if self.selected_transport
            else None,
            "selected_graphs": (
                self.selected_transport.graphs.stats()
                if self.selected_backend
                in ("tilelang_gpu", "tilelang_gr", "tilelang_tma", "triton_gr")
                and self.selected_transport.graphs is not None
                else None
            ),
            "cuda_graphs": self.graphs is not None,
            "graphs": self.graphs.stats() if self.graphs is not None else None,
            "cache_graphs": (
                self.cache.plan_graphs.stats()
                if self.cache.plan_graphs is not None
                else None
            ),
            "compute_graphs": (
                self.compute_graphs.stats() if self.compute_graphs is not None else None
            ),
            "gr_compute_graphs": self.gr_compute.stats()["graphs"]
            if self.gr_compute
            else None,
            "gr_local_graphs": self.gr_local.stats()["graphs"]
            if self.gr_local
            else None,
            "gr_finish_down_projection": "active_q_eager" if self.gr_compute else None,
            "gr_finish_down_calls": self.gr_finish_down_calls,
            "gr_bucket_memory": {
                "compute": self.gr_compute.stats(),
                "local": self.gr_local.stats(),
            }
            if self.gr_compute
            else None,
        }

    def batch_forward_stats(self):
        return [
            dict(row, nsa_execution=self._execution_stats())
            for row in super().batch_forward_stats()
        ]

    def forward_stats(self):
        result = super().forward_stats()
        return dict(result, nsa_execution=self._execution_stats()) if result else result

    def close(self):
        if self.selected_transport is not None:
            self.selected_transport.close()
        if self.cache.plan_graphs is not None:
            self.cache.plan_graphs.close()
        if self.compute_graphs is not None:
            self.compute_graphs.close()
        for executor in (self.gr_compute, self.gr_local):
            if executor is not None:
                executor.close()
        if self.graphs is not None:
            self.graphs.close()
        super().close()

    @classmethod
    def ensure_available(cls):
        try:
            import safetensors  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NSA needs uv sync --locked --group nsa") from exc

    def _load_weights(self, path):
        from safetensors import safe_open

        cfg = self.config
        hidden, intermediate = cfg.hidden_size, cfg.intermediate_size
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = self.heads * cfg.head_dim
        shapes = {
            "model.embed_tokens.weight": (cfg.vocab_size, hidden),
            "model.norm.weight": (hidden,),
        }
        for layer in range(self.layers):
            for name, shape in {
                "input_layernorm.weight": (hidden,),
                "post_attention_layernorm.weight": (hidden,),
                "self_attn.q_proj.weight": (q_dim, hidden),
                "self_attn.k_proj.weight": (kv_dim, hidden),
                "self_attn.v_proj.weight": (kv_dim, hidden),
                "self_attn.o_proj.weight": (hidden, q_dim),
                "self_attn.g_proj.weight": (cfg.num_attention_heads * 3, hidden),
                "mlp.gate_proj.weight": (intermediate, hidden),
                "mlp.up_proj.weight": (intermediate, hidden),
                "mlp.down_proj.weight": (hidden, intermediate),
            }.items():
                shapes[f"model.layers.{layer}.{name}"] = shape
        with safe_open(
            path / "model.safetensors", framework="pt", device="cpu"
        ) as source:
            keys = set(source.keys())
            self.candidate_head_key = (
                "lm_head.weight"
                if "lm_head.weight" in keys
                else "model.embed_tokens.weight"
            )
            if "lm_head.weight" in keys or not cfg.tie_word_embeddings:
                shapes["lm_head.weight"] = (cfg.vocab_size, hidden)
            if set(shapes) != keys:
                raise ValueError(
                    "NSA weight names disagree with checkpoint architecture"
                )
            for name, shape in shapes.items():
                view = source.get_slice(name)
                if tuple(view.get_shape()) != shape or view.get_dtype() != "BF16":
                    raise ValueError(f"Invalid NSA weight shape/dtype: {name}")
            # Keep the full output head off GPU; recommendation loads selected rows.
            return {
                name: source.get_tensor(name).to("cuda")
                for name in shapes
                if name != "lm_head.weight"
            }

    def _empty_user(self, row):
        def empty():
            return torch.empty(
                (0, self.heads, self.dim), device="cuda", dtype=torch.bfloat16
            )

        return {
            "tokens": [],
            "row": row,
            "pages": [[[] for _ in range(self.heads)] for _ in range(self.layers)],
            "summaries": [
                {"compressed": empty(), "tail": empty(), "window": empty()}
                for _ in range(self.layers)
            ],
        }

    def _project(self, hidden, positions, layer):
        cfg = self.config

        def weight(name):
            return self.weights[f"model.layers.{layer}.{name}"]

        normalized = common.rms_norm(
            hidden, weight("input_layernorm.weight"), cfg.rms_norm_eps
        )
        query = F.linear(normalized, weight("self_attn.q_proj.weight")).view(
            -1, cfg.num_attention_heads, cfg.head_dim
        )
        key = F.linear(normalized, weight("self_attn.k_proj.weight")).view(
            -1, self.heads, cfg.head_dim
        )
        value = F.linear(normalized, weight("self_attn.v_proj.weight")).view_as(key)
        gates = (
            F.linear(normalized, weight("self_attn.g_proj.weight"))
            .view(-1, self.heads, self.groups, 3)
            .sigmoid()
        )
        query, key = common.rotary(query, key, positions, self.inv_freq)
        query = query.reshape(-1, self.heads, self.groups, cfg.head_dim)
        records = torch.cat((key, value), -1)
        return query, records, gates

    def _finish_prepare(self, residual, selected, compressed, window, gates, layer):
        def weight(name):
            return self.weights[f"model.layers.{layer}.{name}"]

        # Preserve the upstream BF16 multiply / addcmul rounding sequence.
        mixed = selected * gates[..., 1, None]
        mixed = torch.addcmul(mixed, compressed, gates[..., 0, None])
        mixed = torch.addcmul(mixed, window, gates[..., 2, None])
        hidden = residual + F.linear(
            mixed.flatten(1), weight("self_attn.o_proj.weight")
        )
        normalized = common.rms_norm(
            hidden, weight("post_attention_layernorm.weight"), self.config.rms_norm_eps
        )
        mlp = F.silu(F.linear(normalized, weight("mlp.gate_proj.weight"))) * F.linear(
            normalized, weight("mlp.up_proj.weight")
        )
        return hidden, mlp

    def _finish(self, residual, selected, compressed, window, gates, layer):
        hidden, mlp = self._finish_prepare(
            residual, selected, compressed, window, gates, layer
        )
        return (
            hidden
            + F.linear(mlp, self.weights[f"model.layers.{layer}.mlp.down_proj.weight"]),
        )

    def _compute(self, phase, layer, function, inputs):
        with torch.cuda.nvtx.range(f"nsa.compute.{phase}.layer.{layer}"):
            return self._compute_unmarked(phase, layer, function, inputs)

    def _project_and_prepare(
        self, hidden, positions, compressed, tail, window, *, layer
    ):
        query, records, gates = self._project(hidden, positions, layer=layer)
        summaries, _ = ops.append_compressed(
            compressed, tail, records, self.config.block_size
        )
        return (
            query,
            records,
            gates,
            summaries,
            torch.cat((window, records)),
            *(records[:, head].contiguous() for head in range(self.heads)),
        )

    def _compute_unmarked(self, phase, layer, function, inputs):
        if self._gr_bucket_execution:
            if phase == "tilelang_finish":
                hidden, mlp = self.gr_compute.run(
                    (phase, layer),
                    partial(self._finish_prepare, layer=layer),
                    inputs,
                    active_q=len(inputs[0]),
                )
                # cuBLAS chooses a different split-K reduction for M=105 and
                # padded M=128 at K=8192. Its small BF16 rounding difference
                # changes later Top-K ties. Keep this GEMM at the real M,
                # outside the reusable bucket graph, and preserve the original
                # separately rounded residual addition (do not fuse addmm).
                output = hidden + F.linear(
                    mlp, self.weights[f"model.layers.{layer}.mlp.down_proj.weight"]
                )
                self.gr_compute.mark_consumed(hidden.device)
                self.gr_finish_down_calls += 1
                return (output,)
            options = (
                dict(pad_inputs=(0, 1), fill_values={1: -1}, slice_outputs=())
                if phase == "projection_prepare"
                else {}
            )
            return self.gr_compute.run(
                (phase, layer), function, inputs, active_q=len(inputs[0]), **options
            )
        if self.compute_graphs is None:
            return function(*inputs)
        return self.compute_graphs.run(
            (phase, layer), function, inputs, clone_outputs=False
        )

    def _layer(self, hidden, positions, entries, indptr, layer, timing):
        with torch.cuda.nvtx.range(f"nsa.layer.{layer}"):
            sizes = [b - a for a, b in zip(indptr, indptr[1:])]
            reason = None
            if self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr"):
                if max(sizes) > self.selected_max_queries:
                    reason = "configured_query_limit"
                else:
                    for size, entry in zip(sizes, entries):
                        reason = self.selected_transport.refusal(
                            size,
                            entry["old"],
                            self.heads,
                            self.groups,
                            self.config.block_counts,
                            self.config.block_size,
                        )
                        if reason:
                            break
                self._gr_allowed = reason is None
                if reason:
                    self.selected_fallback_reasons[reason] = (
                        self.selected_fallback_reasons.get(reason, 0) + 1
                    )
            self.selected_request_info = dict(
                effective_q=sizes,
                history=[e["prefix"] for e in entries],
                selected_limit=self.selected_max_queries,
                fallback_reason=reason,
            )
            self._gr_bucket_execution = (
                self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr")
                and self._gr_allowed
                and len(entries) == 1
                and entries[0]["old"] == entries[0]["prefix"]
                and len(hidden) <= min(self.selected_max_queries, 1024)
            )
            try:
                return self._layer_unmarked(
                    hidden, positions, entries, indptr, layer, timing
                )
            finally:
                self._gr_bucket_execution = False

    def _layer_unmarked(self, hidden, positions, entries, indptr, layer, timing):
        cfg = self.config
        timer = StageTimer(timing.synchronize) if timing is not None else None
        residual = hidden
        gpu_blocks = self.selected_backend in (
            "tilelang_gpu",
            "tilelang_gr",
            "tilelang_tma",
            "triton_gr",
        ) and max(b - a for a, b in zip(indptr, indptr[1:])) <= min(
            self.selected_max_queries,
            1024
            if self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr")
            else self.selected_max_queries,
        )
        if self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr"):
            gpu_blocks = gpu_blocks and self._gr_allowed
        prepared = (
            gpu_blocks
            and len(entries) == 1
            and entries[0]["old"] == entries[0]["prefix"]
        )
        if prepared:
            saved = entries[0]["state"]["summaries"][layer]
            query, records, gates, compressed, window, *head_records = self._compute(
                "projection_prepare",
                layer,
                partial(self._project_and_prepare, layer=layer),
                (
                    hidden,
                    positions,
                    saved["compressed"],
                    saved["tail"],
                    saved["window"],
                ),
            )
            if self._gr_bucket_execution:
                query, records, gates = (
                    t[: len(hidden)] for t in (query, records, gates)
                )
                head_records = [t[: len(hidden)] for t in head_records]
        else:
            query, records, gates = self._compute(
                "projection",
                layer,
                partial(self._project, layer=layer),
                (hidden, positions),
            )
        if timer is not None:
            timer.mark("before_kv_write_ms")

        summaries, windows, fetches = [], [], []
        write_slots, write_values = [], []
        for i, entry in enumerate(entries):
            begin, end = indptr[i : i + 2]
            state, old, history = entry["state"], entry["old"], entry["prefix"]
            saved = state["summaries"][layer]
            current = records[begin:end]
            if not prepared:
                compressed, _ = ops.append_compressed(
                    saved["compressed"], saved["tail"], current, cfg.block_size
                )
            summaries.append(compressed)
            if not prepared:
                window = torch.cat((saved["window"], current))
            windows.append((window, old - len(saved["window"])))
            added = history - old
            if added:
                saved["compressed"], saved["tail"] = ops.append_compressed(
                    saved["compressed"], saved["tail"], current[:added], cfg.block_size
                )
                persistent_window = torch.cat((saved["window"], current[:added]))
                saved["window"] = persistent_window[
                    max(0, len(persistent_window) - cfg.window_size + 1) :
                ].clone()
            for head in range(self.heads):
                pages = state["pages"][layer][head]
                slots = self.pool.token_slots(pages, old, history)
                row = state["row"] + layer * self.heads + head
                if slots:
                    self.cache.append_slots(row, old, slots)
                    write_slots.extend(slots)
                    write_values.append(current[:added, head])
                fetches.append(
                    {
                        "row": row,
                        "history_length": history,
                        "cached_length": old,
                        "current_kv": head_records[head]
                        if prepared
                        else current[:, head].contiguous(),
                        "pages": pages,
                    }
                )
        if write_slots:
            payload = torch.cat(write_values).contiguous()
            self.pool.write_many(write_slots, payload)
        if timer is not None:
            timer.mark("kv_write_ms")

        compressed_output, window_output = (
            (None, None)
            if gpu_blocks and len(entries) == 1
            else (torch.empty_like(query), torch.empty_like(query))
        )
        for i, entry in enumerate(entries):
            begin, end = indptr[i : i + 2]
            window, window_start = windows[i]
            local_compressed, local_window, selected = self._local_branches(
                query[begin:end],
                summaries[i],
                window,
                positions[begin:end],
                entry["total"],
                window_start,
                return_blocks=gpu_blocks,
                borrow=gpu_blocks and len(entries) == 1,
            )
            if gpu_blocks and len(entries) == 1:
                compressed_output, window_output = local_compressed, local_window
            else:
                compressed_output[begin:end], window_output[begin:end] = (
                    local_compressed,
                    local_window,
                )
            width = min(
                self.topk,
                max(
                    (e["total"] + cfg.block_size - 1) // cfg.block_size for e in entries
                )
                * cfg.block_size,
            )
            if gpu_blocks:
                width //= cfg.block_size
            if self._gr_bucket_execution:
                width = selected.shape[-1]
            if not gpu_blocks or width != selected.shape[-1]:
                selected = F.pad(selected, (0, width - selected.shape[-1]), value=-1)
            for head in range(self.heads):
                fetches[i * self.heads + head]["indices"] = selected[:, head]
        if timer is not None:
            timer.mark("indexer_and_local_branches_ms")
        use_tilelang = self.selected_transport is not None and max(
            b - a for a, b in zip(indptr, indptr[1:])
        ) <= min(
            self.selected_max_queries,
            1024
            if self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr")
            else self.selected_max_queries,
        )
        if self.selected_backend in ("tilelang_gr", "tilelang_tma", "triton_gr"):
            use_tilelang = use_tilelang and self._gr_allowed
        from cxl_recsys.mempool.device_plan import DeferredLayerCounters

        if use_tilelang:
            outputs, counters, deferred = [], [], []
            for i, entry in enumerate(entries):
                begin, end = indptr[i : i + 2]
                selection = torch.stack(
                    [
                        f["indices"]
                        for f in fetches[i * self.heads : (i + 1) * self.heads]
                    ],
                    dim=1,
                )
                # FLA's block expansion keeps one full BS slot per selected
                # block. A causally invalid block is entirely -1; a tail block
                # always retains its first token, so this recovers block IDs.
                blocks = (
                    selection.to(torch.int32)
                    if gpu_blocks
                    else torch.where(
                        selection[..., :: cfg.block_size] >= 0,
                        selection[..., :: cfg.block_size] // cfg.block_size,
                        -1,
                    ).to(torch.int32)
                )
                selected, counts = self.selected_transport.run(
                    query[begin:end],
                    blocks,
                    positions[begin:end],
                    fetches[i * self.heads : (i + 1) * self.heads],
                    cfg.block_size,
                )
                # Bounded transport buffers are reused by the next user. Own
                # the first output before another same-shape invocation.
                outputs.append(
                    selected.clone() if gpu_blocks and len(entries) > 1 else selected
                )
                if isinstance(counts, DeferredLayerCounters):
                    deferred.append(counts)
                else:
                    counters.extend(counts)
            selected = (
                outputs[0] if gpu_blocks and len(outputs) == 1 else torch.cat(outputs)
            )
            if deferred:
                counters = (
                    deferred[0]
                    if len(deferred) == 1
                    else DeferredLayerCounters(
                        torch.cat([c.statistics for c in deferred]),
                        sum(c.launches for c in deferred),
                        self.pool.backend,
                    )
                )
            self.selected_calls += 1
        else:
            with torch.cuda.nvtx.range("nsa.cache.fetch"):
                batch, counters = self.cache.fetch_layer(
                    [{k: v for k, v in f.items() if k != "pages"} for f in fetches]
                )
            if self.selected_transport is not None:
                self.selected_fallbacks += 1
        if isinstance(counters, DeferredLayerCounters):
            self._pending_cache_stats.append((entries, counters))
        else:
            for i, counter in enumerate(counters):
                for name, count in counter.items():
                    entries[i // self.heads]["stats"][name] += count
        for i, entry in enumerate(entries):
            entry["stats"]["queries"] += indptr[i + 1] - indptr[i]
        if timer is not None:
            timer.mark("kv_read_ms")

        if use_tilelang:
            (hidden,) = self._compute(
                "tilelang_finish",
                layer,
                partial(self._finish, layer=layer),
                (residual, selected, compressed_output, window_output, gates),
            )
            if timer is not None and timing is not None:
                timer.mark("selected_attention_projection_and_ffn_ms")
                timing.layers.append({"layer": layer, **timer.report()})
            return hidden

        def selected_finish(
            residual, query, values, indices, compressed, window, gates
        ):
            packed = torch.cat(
                [
                    query[begin:end, head]
                    for begin, end in zip(indptr, indptr[1:])
                    for head in range(self.heads)
                ]
            )
            if self.selected_backend == "tilelang_indexed":
                from cxl_recsys.models.nsa.backends.indexed import indexed_attention

                attended = indexed_attention(packed, values, indices, cfg.block_size)
            else:
                attended = ops.indexed_attention(
                    packed, values, indices, self.query_tile
                )
            outputs, cursor = [], 0
            for begin, end in zip(indptr, indptr[1:]):
                count = end - begin
                outputs.append(
                    attended[cursor : cursor + count * self.heads]
                    .view(self.heads, count, self.groups, cfg.head_dim)
                    .transpose(0, 1)
                )
                cursor += count * self.heads
            return self._finish(
                residual, torch.cat(outputs), compressed, window, gates, layer
            )

        (hidden,) = self._compute(
            ("selected_finish", tuple(indptr)),
            layer,
            selected_finish,
            (
                residual,
                query,
                batch.values,
                batch.indices,
                compressed_output,
                window_output,
                gates,
            ),
        )
        if timer is not None and timing is not None:
            timer.mark("selected_attention_projection_and_ffn_ms")
            timing.layers.append({"layer": layer, **timer.report()})
        if self.selected_backend == "tilelang_indexed":
            self.selected_calls += 1
        return hidden
