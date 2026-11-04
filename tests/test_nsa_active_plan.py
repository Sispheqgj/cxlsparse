"""Active-plan semantics and compact-capacity bounds independent of attention."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cxl_recsys.models.nsa.planning.active import (
    ActivePlanAdapter,
    reference_active_plan,
)


def _fetch(old=19, history=19, total=25, hot=(), offset=0):
    return dict(
        cached_length=old,
        history_length=history,
        total_length=total,
        hot_locations={token: offset + 100 + i for i, token in enumerate(hot)},
        host_slots={token: offset + token * 3 for token in range(old)},
    )


def test_reference_duplicate_occurrences_partial_hits_and_causal_tail():
    blocks = torch.tensor([[[0, 0, 2]], [[0, 1, 2]]], dtype=torch.int32)
    positions = torch.tensor([19, 24])
    f = _fetch(hot=(0, 2, 16, 18))
    plan = reference_active_plan(blocks, positions, [f], block_size=8, hot=32)
    assert plan.keys == ((0, 0), (0, 2), (0, 1))
    assert plan.required == (8, 8, 8)
    assert plan.row_keys[0] == ((0, 0), (0, 0), (0, 2))
    assert plan.sources[0][0] == ("hot", 100)
    assert plan.sources[0][1] == ("external", 3)
    assert plan.sources[1][:4] == (
        ("hot", 102),
        ("external", 51),
        ("hot", 103),
        ("current", 0),
    )
    assert plan.hot[0] == tuple(range(8)) + (16, 17, 18) + tuple(range(8, 16))
    # 24 unique required tokens; 19 historical; duplicate occurrences count again.
    assert plan.stats[0] == (24, 19, 4, 15, 38, 6, 0)


@pytest.mark.parametrize("mode", ["miss", "hit", "empty"])
def test_reference_empty_and_full_residency(mode):
    blocks = torch.tensor([[[0, 1]], [[0, 2]]], dtype=torch.int32)
    if mode == "empty":
        blocks.fill_(-1)
    f = _fetch(hot=range(19) if mode == "hit" else ())
    plan = reference_active_plan(
        blocks, torch.tensor([19, 24]), [f], block_size=8, hot=7
    )
    assert plan.stats[0][3] == (19 if mode == "miss" else 0)
    assert plan.stats[0][2] == (19 if mode == "hit" else 0)
    if mode == "empty":
        assert not plan.keys and not plan.hot[0]
        assert plan.stats[0] == (0,) * 7
    else:
        assert plan.hot[0] == tuple(range(7))


def test_reference_padded_rows_do_not_select_or_mutate_hot():
    blocks = torch.tensor([[[1, 1]], [[-1, -1]], [[0, 2]]], dtype=torch.int32)
    f = _fetch()
    actual = reference_active_plan(
        blocks, torch.tensor([19, -1, -1]), [f], block_size=8, hot=32
    )
    expected = reference_active_plan(
        blocks[:1], torch.tensor([19]), [f], block_size=8, hot=32
    )
    assert actual == replace(expected, row_keys=expected.row_keys + ((), ()))


def test_reference_owner_heads_append_and_backing_replacement():
    blocks = torch.tensor([[[2], [2]]], dtype=torch.int32)
    first = _fetch(hot=(16,))
    second = _fetch(offset=1000)
    plan = reference_active_plan(
        blocks, torch.tensor([24]), [first, second], block_size=8, hot=32
    )
    assert plan.keys == ((0, 2), (1, 2))
    assert plan.sources[0][0] == ("hot", 100)
    assert plan.sources[1][0] == ("external", 1048)
    replacement = _fetch(old=23, history=23, total=29, hot=(21,), offset=2000)
    replay = reference_active_plan(
        blocks[:, :1], torch.tensor([24]), [replacement], block_size=8, hot=32
    )
    assert replay.sources[0][0] == ("external", 2048)
    assert replay.sources[0][5] == ("hot", 2100)
    assert replay.sources[0][7] == ("current", 0)
    assert replay.hot[0] == tuple(range(16, 23))


def test_reference_capacity_overflow_and_invalid_selection_fail():
    with pytest.raises(ValueError, match="overflow"):
        reference_active_plan(
            torch.tensor([[[0, 1, 2]]]),
            torch.tensor([24]),
            [_fetch()],
            block_size=8,
            hot=8,
            capacity=2,
        )
    with pytest.raises(ValueError, match="Invalid selected"):
        reference_active_plan(
            torch.tensor([[[4]]]), torch.tensor([24]), [_fetch()], block_size=8, hot=8
        )


def _adapter(backend="cxl"):
    return ActivePlanAdapter(
        SimpleNamespace(hot=256, pool=SimpleNamespace(backend=backend))
    )


@pytest.mark.parametrize("q", [64, 128, 256, 512, 1024])
def test_geometry_capacity_never_truncates_union(q):
    query = torch.empty((q, 2, 16, 64), device="meta")
    blocks = torch.empty((q, 2, 4), device="meta", dtype=torch.int32)
    f = dict(cached_length=65536, current_kv=range(q))
    geom = _adapter().geometry(query, blocks, [f, f], 16, map_blocks=4160)
    assert geom["per_head"] == min(4160, q * 4)
    assert geom["map_stride"] == 4161
    assert geom["physical"] == geom["per_head"] * 2
    assert geom["extent"] == geom["per_head"] * 16
    assert geom["sort_extent"] >= 256


def test_fixed_bucket_shape_and_long_history_mapping_only():
    query = torch.empty((64, 2, 16, 64), device="meta")
    blocks = torch.empty((64, 2, 4), device="meta", dtype=torch.int32)
    observed = []
    for history in (4096, 8192, 16384, 32768, 65536):
        fetch = dict(cached_length=history, current_kv=range(64))
        observed.append(_adapter().geometry(query, blocks, [fetch] * 2, 16))
    for geom in observed:
        assert geom["per_head"] == 256
        assert geom["extent"] == 4096
    assert [g["map_blocks"] for g in observed] == [260, 516, 1028, 2052, 4100]
    fetch = dict(cached_length=4096, current_kv=range(64))
    with pytest.raises(ValueError, match="truncate"):
        _adapter().geometry(query, blocks, [fetch] * 2, 16, map_blocks=256)


@pytest.mark.gpu
@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
def test_gpu_compact_plan_replay_sources_hot_and_changed_owner(residency):
    from test_nsa_transport import setup_case

    from cxl_recsys.models.nsa.planning.active import _copy_hbm

    pool, cache, backing, query, positions, blocks, fetches = setup_case(residency)
    adapter = ActivePlanAdapter(cache, chunk_blocks=3)
    padded_query = torch.zeros(
        (64, *query.shape[1:]), dtype=query.dtype, device=query.device
    )
    padded_blocks = torch.full(
        (64, *blocks.shape[1:]), -1, dtype=blocks.dtype, device=blocks.device
    )
    padded_positions = torch.full(
        (64,), -1, dtype=positions.dtype, device=positions.device
    )
    padded_query[: len(query)] = query
    padded_blocks[: len(blocks)] = blocks
    padded_positions[: len(positions)] = positions
    w = adapter.allocate(padded_query, padded_blocks, fetches, 16, map_blocks=64)
    bases = torch.tensor(
        [pool._cuda_mapping(query.device).address],
        dtype=torch.int64,
        device=query.device,
    )
    try:
        for repeat in range(3):
            if repeat == 2:
                # Same workspace, another owner/head layout and changed backing map.
                fetches = list(reversed(fetches))
            desc = torch.tensor(
                [
                    (
                        f["row"],
                        f["cached_length"],
                        f["history_length"],
                        f["current_kv"].data_ptr(),
                        len(f["current_kv"]),
                    )
                    for f in fetches
                ],
                dtype=torch.int64,
                device=query.device,
            )
            cpu_fetches = []
            for f in fetches:
                row = f["row"]
                hot = cache.tags[row, : cache.hot].cpu().tolist()
                loc = cache.locs[row, : cache.hot].cpu().tolist()
                cpu_fetches.append(
                    dict(
                        cached_length=67,
                        history_length=67,
                        total_length=80,
                        hot_locations={
                            token: location
                            for token, location in zip(hot, loc, strict=True)
                            if token >= 0
                        },
                        host_slots=dict(
                            enumerate(cache.host_locs[row, :67].cpu().tolist())
                        ),
                    )
                )
            expected = reference_active_plan(
                padded_blocks.cpu(),
                padded_positions.cpu(),
                cpu_fetches,
                block_size=16,
                hot=cache.hot,
            )
            w["kv"].fill_(123)
            selected = adapter.plan(padded_blocks, padded_positions, desc, bases, w)
            torch.cuda.synchronize()
            assert w["stats"].cpu().tolist() == [list(row) for row in expected.stats]
            assert w["all_counts"][len(query) * 2 :].eq(0).all().item()
            assert w["active_counts"].sum().item() == len(expected.keys)
            for head in range(2):
                count = int(w["active_counts"][head].item())
                keys = w["active_keys"][head, :count].cpu().tolist()
                assert keys == sorted(b for h, b in expected.keys if h == head)
                assert w["mapping"][head, keys].cpu().tolist() == list(range(count))
            expected_stage = torch.zeros_like(w["kv"], device="cpu")
            source_map = dict(zip(expected.keys, expected.sources, strict=True))
            for head, f in enumerate(fetches):
                count = int(w["active_counts"][head].item())
                keys = w["active_keys"][head, :count].cpu().tolist()
                for compact, block in enumerate(keys):
                    physical = head * w["per_head"] + compact
                    for offset, (kind, source) in enumerate(source_map[head, block]):
                        if kind == "hot":
                            expected_stage[physical, offset] = cache.data[source].cpu()
                        elif kind == "current":
                            expected_stage[physical, offset] = f["current_kv"][
                                source
                            ].cpu()
            torch.testing.assert_close(w["kv"].cpu(), expected_stage, rtol=0, atol=0)
            # Read the actual chosen per-token source pointers, including partial hits.
            _copy_hbm[(w["physical"] * 16,)](w["address"], w["kv"], w["physical"] * 16)
            adapter.retain_hot(desc, selected, w)
            torch.cuda.synchronize()
            for head, f in enumerate(fetches):
                expected_hot = list(expected.hot[head])
                tags = cache.tags[f["row"], : cache.hot].cpu()
                assert tags[tags >= 0].tolist() == expected_hot
                if expected_hot:
                    actual = cache.data[
                        cache.locs[f["row"], : len(expected_hot)].long()
                    ].cpu()
                    torch.testing.assert_close(
                        actual, backing[expected_hot, f["row"]], rtol=0, atol=0
                    )
    finally:
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("tile", [1, 8, 16])
def test_gpu_tiled_hbm_copy_preserves_null_sources_tail_and_guards(tile):
    import triton

    from cxl_recsys.models.nsa.planning.active import _copy_hbm

    tokens = 13
    source = torch.arange(tokens * 128, device="cuda").reshape(tokens, 128).bfloat16()
    indices = [12, -1, 3, -1, -1, 0, -1, 7, 8, -1, -1, 2, 2]
    pointers = torch.tensor(
        [0 if i < 0 else source[i].data_ptr() for i in indices],
        device="cuda",
        dtype=torch.int64,
    )
    destination = torch.full(
        (tokens + 2, 128), 123, dtype=torch.bfloat16, device="cuda"
    )
    _copy_hbm[(triton.cdiv(tokens, tile),)](pointers, destination, tokens, TOKENS=tile)
    expected = torch.stack(
        [torch.zeros_like(source[0]) if i < 0 else source[i] for i in indices]
    )
    torch.testing.assert_close(destination[:tokens], expected, rtol=0, atol=0)
    assert destination[tokens:].eq(123).all().item()


def test_all_hbm_consumer_does_not_allocate_unused_chunk_stages():
    query = torch.empty((1024, 2, 16, 64), device="meta")
    blocks = torch.empty((1024, 2, 16), device="meta", dtype=torch.int32)
    f = dict(cached_length=65536, current_kv=range(1024))
    streamed = _adapter().geometry(query, blocks, [f, f], 16, map_blocks=8192)
    ready = _adapter().geometry(
        query, blocks, [f, f], 16, map_blocks=8192, all_hbm=True
    )
    assert streamed["stages"] == 257
    assert ready["stages"] == 2
    assert ready["physical"] == streamed["physical"]
    assert ready["chunk"] == ready["physical"]


@pytest.mark.gpu
def test_staged_occurrences_preserve_original_online_softmax_order():
    """Resident and early-ready blocks cannot overtake an earlier selection."""
    from cxl_recsys.models.nsa.planning.active import _stages

    blocks = torch.tensor(
        [[[4, 3, 1, 2, 0]], [[4, 3, 1, 2, 0]]], dtype=torch.int32, device="cuda"
    )
    positions = torch.tensor([79, -1], dtype=torch.int32, device="cuda")
    mapping = torch.arange(6, dtype=torch.int32, device="cuda")
    flags = torch.tensor([1, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
    ranks = torch.tensor([4, 0, 0, 2, 0], dtype=torch.int32, device="cuda")
    stats = torch.zeros(7, dtype=torch.int64, device="cuda")
    ids = torch.full((4, 2, 5), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros((4, 2), dtype=torch.int32, device="cuda")
    row_positions = torch.empty(2, dtype=torch.int32, device="cuda")
    all_ids = torch.full((2, 5), -1, dtype=torch.int32, device="cuda")
    all_counts = torch.zeros(2, dtype=torch.int32, device="cuda")
    _stages[(2, 4)](
        blocks,
        positions,
        mapping,
        flags,
        ranks,
        stats,
        ids,
        counts,
        row_positions,
        all_ids,
        all_counts,
        1,
        2,
        5,
        8,
        5,
        5,
        16,
        2,
        num_warps=1,
    )
    actual_ids, actual_counts = ids.cpu(), counts.cpu()
    ordered = [
        int(block)
        for stage in range(4)
        for block in actual_ids[stage, 0, : actual_counts[stage, 0]]
    ]
    assert ordered == all_ids[0, : all_counts[0]].cpu().tolist() == [4, 3, 1, 2, 0]
    assert actual_counts[:, 0].tolist() == [1, 0, 3, 1]
    assert actual_counts[:, 1].eq(0).all()
    assert all_counts.cpu().tolist() == [5, 0]
