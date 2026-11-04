"""Bucket boundaries, scratch ownership and masked padding for GR execution."""

import os

import pytest
import torch
from test_nsa import checkpoint  # noqa: F401

from cxl_recsys.models.nsa.dispatch import (
    BucketCapacityError,
    BucketOwner,
    NSABucketExecutor,
    NSABucketInputs,
    block_bucket,
    consumer_groups,
    query_bucket,
)


@pytest.mark.parametrize("group", [2, 4, 8])
def test_unaccepted_shared_consumer_rejected_before_checkpoint_or_gpu(group):
    from cxl_recsys.models.nsa import NSAModel

    with pytest.raises(ValueError, match="full-model numerical validation failed"):
        NSAModel(
            None,
            model_path="missing-checkpoint-must-not-be-opened",
            selected_backend="tilelang_gr",
            selected_query_group=group,
        )


@pytest.mark.parametrize(
    "active,capacity",
    [
        (1, 64),
        (63, 64),
        (64, 64),
        (65, 128),
        (105, 128),
        (127, 128),
        (128, 128),
        (129, 256),
        (255, 256),
        (256, 256),
        (257, 512),
        (512, 512),
        (1024, 1024),
    ],
)
def test_finite_query_buckets(active, capacity):
    assert query_bucket(active) == capacity
    groups = consumer_groups(capacity)
    covered = [q for first, last in groups for q in range(first, last)]
    assert covered == list(range(capacity))
    assert all(last - first <= 128 for first, last in groups)


@pytest.mark.parametrize("active", [0, -1, 1025, 4096])
def test_out_of_range_queries_require_explicit_fallback(active):
    with pytest.raises(BucketCapacityError):
        query_bucket(active)


def test_active_block_capacity_never_truncates():
    assert block_bucket(0, maximum=48) == 48
    assert block_bucket(65, maximum=1000) == 128
    assert block_bucket(1000, maximum=1000) == 1000
    with pytest.raises(BucketCapacityError):
        block_bucket(1001, maximum=1000)


def _selection(q, *, value=1, device="cpu"):
    query = torch.full((q, 2, 4, 8), float(value), device=device)
    blocks = torch.full((q, 2, 3), value, dtype=torch.int32, device=device)
    positions = torch.arange(8192, 8192 + q, dtype=torch.int64, device=device)
    return query, blocks, positions


def test_selected_mixed_users_clear_inactive_rows_and_reuse_addresses():
    storage = NSABucketInputs()
    try:
        first = storage.prepare(
            *_selection(127), owner=BucketOwner("user-a", 0, "pool-a", 1)
        )
        key = first.key
        addresses = tuple(
            t.data_ptr() for t in (first.query, first.blocks, first.positions)
        )
        with pytest.raises(RuntimeError, match="still leased"):
            storage.prepare(
                *_selection(105), owner=BucketOwner("user-b", 1, "pool-b", 2)
            )
        first.release()
        with storage.prepare(
            *_selection(65, value=7),
            owner=BucketOwner("user-b", 1, "pool-b", 2, memory_version=3),
        ) as second:
            assert second.key == key
            assert (
                tuple(
                    t.data_ptr()
                    for t in (second.query, second.blocks, second.positions)
                )
                == addresses
            )
            assert second.blocks[:65].eq(7).all()
            assert second.blocks[65:].eq(-1).all()
            assert second.positions[65:].eq(-1).all()
            assert second.query[65:].eq(0).all()
            with pytest.raises(RuntimeError, match="released or overwritten"):
                first.assert_valid()
        assert storage.stats()["allocations"] == 1
        assert storage.stats()["memory_version"] == 3
        with pytest.raises(RuntimeError):
            second.release()
    finally:
        storage.close()


def test_leases_prevent_close_and_eviction():
    storage = NSABucketInputs(capacity=1)
    first = storage.prepare(*_selection(64), owner=BucketOwner("a", 0, "pool", 1))
    with pytest.raises(RuntimeError, match="live input leases"):
        storage.close()
    with pytest.raises(BucketCapacityError, match="leased"):
        storage.prepare(*_selection(65), owner=BucketOwner("b", 0, "pool", 2))
    first.release()
    with storage.prepare(*_selection(65), owner=BucketOwner("b", 0, "pool", 2)):
        pass
    assert storage.stats()["evictions"] == 1
    storage.close()
    assert storage.stats()["buffer_bytes"] == 0


def test_buffer_budget_is_checked_before_allocation():
    storage = NSABucketInputs(max_bytes=100)
    with pytest.raises(BucketCapacityError, match="input bytes"):
        storage.prepare(*_selection(64), owner=BucketOwner("a", 0, "pool", 1))
    assert storage.stats()["allocations"] == 0
    storage.close()


def test_projection_prepare_padding_preserves_causal_summaries_and_window():
    executor = NSABucketExecutor(cuda_graphs=False)
    block_size, history = 16, 8197
    tail = torch.randn(history % block_size, 2)
    previous = torch.randn(history // block_size, 2)
    window_tail = torch.randn(31, 2)

    def prepare(hidden, positions, compressed, tail, window):
        records = hidden * 2
        values = torch.cat((tail, records))
        complete = len(values) // block_size
        summaries = torch.cat(
            (
                compressed,
                values[: complete * block_size].view(complete, block_size, 2).mean(1),
            )
        )
        return records, positions, summaries, torch.cat((window, records))

    try:
        for active in (127, 65, 105, 128):
            hidden = torch.randn(active, 2)
            positions = torch.arange(history, history + active)
            expected = prepare(hidden, positions, previous, tail, window_tail)
            records, padded_positions, summaries, window = executor.run(
                "prepare",
                prepare,
                (hidden, positions, previous, tail, window_tail),
                active_q=active,
                pad_inputs=(0, 1),
                fill_values={1: -1},
                slice_outputs=(),
            )
            assert records.shape[0] == 128
            assert padded_positions[active:].eq(-1).all()
            torch.testing.assert_close(records[:active], expected[0])
            torch.testing.assert_close(summaries[: len(expected[2])], expected[2])
            torch.testing.assert_close(window[: len(expected[3])], expected[3])
            # Every newly appended summary that contains padding lies strictly
            # beyond the last real query's complete-block causal boundary.
            extra = torch.arange(len(expected[2]), len(summaries))
            assert ((extra + 1) * block_size > positions[-1] + 1).all()
        assert executor.stats()["allocations"] == 1
    finally:
        executor.close()


def test_pure_finish_matches_unpadded_and_clone_outputs_survive_next_request():
    executor = NSABucketExecutor(cuda_graphs=False)
    weight = torch.randn(8, 8)

    def finish(residual, selected, gates):
        return ((residual + selected * gates) @ weight,)

    try:
        for q in (65, 127, 105, 128):
            tensors = tuple(torch.randn(q, 8) for _ in range(3))
            (out,) = executor.run(
                "finish", finish, tensors, active_q=q, clone_outputs=True
            )
            torch.testing.assert_close(out, finish(*tensors)[0])
            saved = out.clone()
            executor.run("finish", finish, tensors, active_q=q)
            torch.testing.assert_close(out, saved)
        assert executor.stats()["allocations"] == 1
        assert executor.stats()["graphs"]["captures"] == 0
    finally:
        executor.close()


def test_model_finish_keeps_down_projection_at_active_rows(monkeypatch):
    from functools import partial
    from types import SimpleNamespace

    from cxl_recsys.models.nsa import NSAModel

    model = NSAModel.__new__(NSAModel)
    model.config = SimpleNamespace(rms_norm_eps=1e-5)
    model.weights = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(24, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(24, 8),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 24),
    }
    model._gr_bucket_execution = True
    model.gr_finish_down_calls = 0
    model.gr_compute = NSABucketExecutor(cuda_graphs=False)
    down = model.weights["model.layers.0.mlp.down_proj.weight"]
    actual_rows = []
    original_linear = torch.nn.functional.linear

    def linear(input, weight, bias=None):
        if weight is down:
            actual_rows.append(len(input))
        return original_linear(input, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    lengths = (63, 65, 105, 127, 128, 129, 255, 257)
    try:
        for q in lengths:
            inputs = (
                torch.randn(q, 8),
                *(torch.randn(q, 2, 2, 2) for _ in range(3)),
                torch.rand(q, 2, 2, 3),
            )
            expected = model._finish(*inputs, layer=0)[0]
            actual_rows.clear()
            (actual,) = model._compute_unmarked(
                "tilelang_finish", 0, partial(model._finish, layer=0), inputs
            )
            assert actual_rows == [q]
            torch.testing.assert_close(actual, expected)
        assert model.gr_finish_down_calls == len(lengths)
        assert model.gr_compute.stats()["allocations"] == 4
    finally:
        model.gr_compute.close()


def test_distinct_layers_share_staging_and_keep_their_own_outputs():
    executor = NSABucketExecutor(cuda_graphs=False)
    try:
        x = torch.randn(105, 8)
        result = []
        for layer, scale in enumerate((2.0, 3.0, -1.0)):
            (out,) = executor.run(
                ("projection", layer),
                lambda value, scale=scale: (value * scale,),
                (x,),
                active_q=len(x),
                clone_outputs=True,
            )
            result.append(out)
        assert executor.stats()["allocations"] == 1
        for out, scale in zip(result, (2.0, 3.0, -1.0)):
            torch.testing.assert_close(out, x * scale)
    finally:
        executor.close()


def test_eager_views_do_not_alias_shared_staging_of_another_phase():
    executor = NSABucketExecutor(cuda_graphs=False)
    try:
        first = torch.arange(105 * 8).reshape(105, 8)
        (out,) = executor.run(
            "identity-view", lambda x: (x[:, 1:],), (first,), active_q=105
        )
        executor.run(
            "another-phase",
            lambda x: (x,),
            (torch.full((65, 8), -7),),
            active_q=65,
        )
        torch.testing.assert_close(out, first[:, 1:])
        assert executor.stats()["allocations"] == 1
    finally:
        executor.close()


@pytest.mark.gpu
@pytest.mark.parametrize("direct", [False, True])
def test_cuda_graph_same_bucket_does_not_recapture_and_stream_handoff(direct):
    executor = NSABucketExecutor(capacity=2, direct_replay=direct)
    device = torch.device("cuda")
    try:
        for q in (63, 64, 65, 105, 127, 128):
            x = torch.arange(q * 8, device=device).reshape(q, 8).float()
            (out,) = executor.run(
                "pointwise", lambda value: (value * 3,), (x,), active_q=q
            )
            torch.testing.assert_close(out, x * 3)
        assert executor.stats()["graphs"]["captures"] == 2
        assert executor.stats()["graphs"]["replays"] == 6
        original = torch.cuda.current_stream()
        consumer = torch.cuda.Stream()
        consumer.wait_stream(original)
        with torch.cuda.stream(consumer):
            saved = out.clone()
            executor.mark_consumed(device)
        x = torch.ones((65, 8), device=device)
        (out,) = executor.run(
            "pointwise", lambda value: (value * 3,), (x,), active_q=65
        )
        original.wait_stream(consumer)
        torch.testing.assert_close(out, x * 3)
        torch.testing.assert_close(
            saved, torch.arange(128 * 8, device=device).reshape(128, 8).float() * 3
        )
    finally:
        executor.close()


@pytest.mark.gpu
@pytest.mark.parametrize("direct", [False, True])
def test_cuda_graph_cross_layer_staging_keeps_weight_keys(direct):
    executor = NSABucketExecutor(capacity=2, direct_replay=direct)
    try:
        x = torch.ones((105, 8), device="cuda")
        outputs = []
        for layer, scale in enumerate((2.0, 3.0)):
            outputs.extend(
                executor.run(
                    ("projection", layer),
                    lambda value, scale=scale: (value * scale,),
                    (x,),
                    active_q=len(x),
                )
            )
        torch.testing.assert_close(outputs[0], x * 2)
        torch.testing.assert_close(outputs[1], x * 3)
        assert executor.stats()["allocations"] == 1
        assert executor.stats()["graphs"]["captures"] == 2
        executor.run(
            ("projection", 0),
            lambda value: (value * 2,),
            (x * 7,),
            active_q=len(x),
        )
        # The first borrowed result expires on its own graph's next replay;
        # the second layer's result remains independently owned by its graph.
        torch.testing.assert_close(outputs[0], x * 14)
        torch.testing.assert_close(outputs[1], x * 3)
        assert executor.stats()["graphs"]["captures"] == 2
        assert executor.stats()["graph_tensor_bytes"] > 0
        executor.graphs.close()
        assert executor.stats()["graph_tensor_bytes"] == 0
    finally:
        executor.close()


@pytest.mark.gpu
def test_direct_replay_padding_refresh_static_inputs_eviction_and_owned_outputs():
    executor = NSABucketExecutor(capacity=2, direct_replay=True)

    def function(value, positions, bias):
        return value * bias, positions, value

    owned = None
    try:
        for step, q in enumerate((127, 65, 105, 128, 17, 63, 129, 65, 105)):
            x = torch.full((q, 8), float(step + 1), device="cuda")
            pos = torch.arange(q, device="cuda", dtype=torch.int64) + 8192
            bias = torch.tensor(float(step + 2), device="cuda")
            product, padded_pos, padded_query = executor.run(
                "mix",
                function,
                (x, pos, bias),
                active_q=q,
                pad_inputs=(0, 1),
                fill_values={1: -1},
                slice_outputs=(),
                clone_outputs=step == 0,
            )
            if step == 0:
                owned = product
            torch.testing.assert_close(product[:q], x * bias)
            torch.testing.assert_close(padded_pos[:q], pos)
            assert padded_pos[q:].eq(-1).all()
            assert padded_query[q:].eq(0).all()
        torch.testing.assert_close(
            owned[:127], torch.full((127, 8), 2.0, device="cuda")
        )
        assert executor.stats()["graphs"]["captures"] == 4
        assert executor.stats()["graphs"]["replays"] == 9
        assert executor.stats()["direct_replays"] == 5
    finally:
        executor.close()
    assert executor.stats()["graph_tensor_bytes"] == 0


@pytest.mark.gpu
def test_gr_model_variable_revisit_graphs_and_history_isolation(checkpoint):  # noqa: F811
    """Real model wiring, without claiming anonymous backing is physical CXL."""
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.base import CacheContext
    from cxl_recsys.models.nsa import NSAModel

    path, cfg, _, _ = checkpoint
    models = []
    try:
        for backend in ("tilelang_gpu", "tilelang_gr"):
            pool = MemoryPool(
                cfg.required_pages(2, 256, 64),
                128,
                kernel_dir=os.environ.get(
                    "NSA_KERNEL_DIR", "artifacts/kernels-tilelang"
                ),
            )
            models.append(
                NSAModel(
                    pool,
                    max_users=2,
                    max_history=256,
                    layers=cfg.num_hidden_layers,
                    model_path=path,
                    selected_backend=backend,
                    selected_max_queries=1024 if backend == "tilelang_gr" else 128,
                    graph_cache_size=8,
                    cuda_graphs=True,
                )
            )

        def run(user, history, extend):
            offset = 9 if user == "a" else 43
            ids = [(i * 7 + offset) % 255 for i in range(history + extend)]
            outputs = [
                m.forward(
                    ids,
                    list(range(len(ids))),
                    cache=CacheContext(user, history),
                ).cpu()
                for m in models
            ]
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0.03, atol=0.03)

        # Establish independent users, then warm the shared 128-query bucket.
        for user in ("a", "b"):
            run(user, 80, 65)
            run(user, 80, 65)
        gr = models[1]

        def captures():
            return (
                gr.gr_compute.graphs.captures,
                gr.gr_local.graphs.captures,
                gr.selected_transport.graphs.captures,
            )

        before = captures()
        for user, q in (("a", 105), ("b", 127), ("a", 128), ("b", 65)):
            down_calls = gr.gr_finish_down_calls
            run(user, 80, q)
            assert captures() == before
            assert gr.gr_finish_down_calls - down_calls == cfg.num_hidden_layers
            assert (
                gr._execution_stats()["gr_finish_down_projection"] == "active_q_eager"
            )
            assert gr.gr_compute.consumer_event is not None
            state = gr.users[user]
            assert all(
                len(s["compressed"]) == 80 // cfg.block_size
                and len(s["tail"]) == 80 % cfg.block_size
                for s in state["summaries"]
            )
            first = state["row"]
            tags = gr.cache.tags[first : first + cfg.num_hidden_layers * 2]
            assert ((tags < 80) | (tags == -1)).all()

        # Each long length must take the new path, including the 128 boundary.
        for q in (129, 255, 257, 512, 1024):
            previous_calls = gr.selected_calls
            previous_fallbacks = gr.selected_fallbacks
            run("a", 80, q)
            assert gr.selected_calls - previous_calls == cfg.num_hidden_layers
            assert gr.selected_fallbacks == previous_fallbacks
            assert gr.selected_transport.last_stats["effective_q"] == q
            assert gr.selected_transport.last_stats["query_bucket"] == query_bucket(q)
            assert gr.selected_transport.last_stats["io_policy"] == "union_once"
            assert (
                gr.selected_transport.last_stats["consumer"] == "native_indexed_bounded"
            )

        # Append is deliberately outside the pure bucketed model segments.
        run("a", 137, 65)
        run("a", 137, 65)
        before = captures()
        run("a", 137, 105)
        assert captures() == before
        assert all(
            len(s["compressed"]) == 137 // cfg.block_size
            and len(s["tail"]) == 137 % cfg.block_size
            for s in gr.users["a"]["summaries"]
        )
    finally:
        for model in models:
            model.close()


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [2, 4])
def test_gr_packed_revisits_keep_user_history_and_hot_sets_isolated(
    checkpoint,  # noqa: F811
    batch_size,
):
    """Exercise actual packed forwards; anonymous backing is not physical CXL."""
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.base import CacheContext
    from cxl_recsys.models.nsa import NSAModel

    path, cfg, _, _ = checkpoint
    histories = [80, 137, 97, 155][:batch_size]
    models = []

    def tokens(user, count):
        return [(index * 13 + user * 41 + 7) % 255 for index in range(count)]

    try:
        for backend in ("tilelang_gpu", "tilelang_gr"):
            pool = MemoryPool(
                cfg.required_pages(batch_size, 256, 64),
                128,
                kernel_dir=os.environ.get(
                    "NSA_KERNEL_DIR", "artifacts/kernels-tilelang"
                ),
            )
            models.append(
                NSAModel(
                    pool,
                    max_users=batch_size,
                    max_history=256,
                    layers=cfg.num_hidden_layers,
                    model_path=path,
                    selected_backend=backend,
                    selected_max_queries=1024 if backend == "tilelang_gr" else 128,
                    graph_cache_size=8,
                    cuda_graphs=True,
                )
            )

        def run(lengths, *, reverse=False, reuse=True):
            order = list(range(batch_size))
            if reverse:
                order.reverse()
            ids, positions, indptr, contexts = [], [], [0], []
            for user in order:
                count = histories[user] + lengths[user]
                ids.extend(tokens(user, count))
                positions.extend(range(count))
                indptr.append(len(ids))
                contexts.append(CacheContext(f"user-{user}", histories[user]))
            outputs = [
                model.forward(ids, positions, seq_indptr=indptr, caches=contexts).cpu()
                for model in models
            ]
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0.03, atol=0.03)
            if reuse:
                assert len(outputs[1]) == sum(lengths)
                for model in models:
                    stats = model.batch_forward_stats()
                    assert [item["input_tokens"] for item in stats] == [
                        lengths[user] for user in order
                    ]
                    assert all(item["mode"] == "reuse" for item in stats)
                    assert all(item["batch"]["size"] == batch_size for item in stats)
            return outputs

        # Persist distinct user prefixes through one actual packed request.
        run([65] * batch_size, reuse=False)
        gr = models[1]
        saved = {
            user: [
                {name: value.clone() for name, value in summary.items()}
                for summary in gr.users[f"user-{user}"]["summaries"]
            ]
            for user in range(batch_size)
        }
        write_bytes = gr.pool.write_bytes
        lengths = (
            (65, 105, 127, 65),
            (127, 65, 105, 127),
            (129, 65, 255, 105),
            (65, 255, 105, 257),
            (257, 105, 129, 65),
            (65, 105, 127, 65),
        )
        for visit, values in enumerate(lengths):
            calls, fallbacks = gr.selected_transport.calls, gr.selected_fallbacks
            run(values[:batch_size], reverse=bool(visit % 2))
            assert (
                gr.selected_transport.calls - calls
                == cfg.num_hidden_layers * batch_size
            )
            assert gr.selected_fallbacks == fallbacks
            assert gr.pool.write_bytes == write_bytes
            seen_rows, seen_pages = set(), set()
            for user, history in enumerate(histories):
                state = gr.users[f"user-{user}"]
                assert state["tokens"] == tokens(user, history)
                first = state["row"]
                rows = set(range(first, first + cfg.num_hidden_layers * 2))
                assert not seen_rows.intersection(rows)
                seen_rows.update(rows)
                pages = {
                    page for layer in state["pages"] for head in layer for page in head
                }
                assert not seen_pages.intersection(pages)
                seen_pages.update(pages)
                tags = gr.cache.tags[first : first + cfg.num_hidden_layers * 2]
                assert ((tags >= 0) & (tags < history) | (tags == -1)).all()
                for before, after in zip(saved[user], state["summaries"]):
                    for name in before:
                        torch.testing.assert_close(
                            after[name], before[name], rtol=0, atol=0
                        )
            # Packed requests use exact-shape model segments. Only transport
            # buckets are shared; this test does not claim packed model reuse.
            assert gr.gr_compute.calls == gr.gr_local.calls == 0
    finally:
        for model in models:
            model.close()
