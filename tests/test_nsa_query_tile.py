"""Membership/duplicate/causal invariants for the optional shared KV consumer."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cxl_recsys.models.nsa.experimental.query_tile import (
    allocate_workspace,
    consume_grouped,
    group_queries,
    query_tile_attention,
    reuse_statistics,
    selected_reference,
)


def make_case(overlap, queries=9, heads=2, width=4, groups=16):
    generator = torch.Generator().manual_seed(89)
    # Physical slots include the KV head. Starts are logical positions and may
    # be identical across heads; physical sharing is confined to one head.
    physical = queries * heads * width
    kv = torch.randn(physical, 16, 128, generator=generator).bfloat16()
    q = torch.randn(queries, heads, groups, 64, generator=generator).bfloat16()
    starts = (torch.arange(physical) % width * 16).int()
    ids = torch.empty((queries * heads, width), dtype=torch.int32)
    for query in range(queries):
        for head in range(heads):
            for slot in range(width):
                shared = overlap == "high" or (overlap == "medium" and slot < 2)
                ids[query * heads + head, slot] = (
                    head if shared else query * heads + head
                ) * width + slot
    counts = torch.full((queries * heads,), width, dtype=torch.int32)
    counts[-1] = 0
    ids[0, 1] = ids[0, 0]  # Duplicate must affect softmax relative weights.
    positions = torch.arange(queries, dtype=torch.int32).repeat_interleave(heads) + 49
    for block in range(physical):
        kv[block, starts[block] + torch.arange(16) > positions.max()] = torch.nan
    return q, kv, ids, starts, positions, counts


def test_logical_reuse_separates_occurrences_and_heads():
    ids = torch.tensor([[2, 2, 4], [20, 20, 40], [2, 4, 8], [20, 40, 80]])
    counts = torch.tensor([3, 3, 3, 3])
    result = reuse_statistics(ids, counts, heads=2, tile_queries=2)
    assert result["weighted_reuse_ratio"] == 5 / 3
    assert result["occurrences"] == 12
    assert result["duplicates"] == 2
    assert result["union_block_loads"] == 6
    assert [t["union_blocks"] for t in result["tiles"]] == [3, 3]


def test_reuse_only_counts_valid_prefix_and_handles_empty():
    ids = torch.tensor([[0, 999], [-1, 888], [0, 777]], dtype=torch.int32)
    counts = torch.tensor([1, 0, 1], dtype=torch.int32)
    stats = reuse_statistics(ids, counts, 1, 2)
    assert stats["occurrences"] == 2
    assert stats["union_block_loads"] == 2
    assert stats["weighted_reuse_ratio"] == 1
    empty = reuse_statistics(ids, torch.zeros_like(counts), 1, 8)
    assert empty["weighted_reuse_ratio"] == 0
    assert empty["union_block_loads"] == 0
    with pytest.raises(ValueError, match="valid counts"):
        reuse_statistics(ids, counts + 3, 1, 2)


def _interpret_grouping_cases():
    import triton

    from cxl_recsys.models.nsa.experimental.query_tile import _membership, _union

    rng = torch.Generator().manual_seed(947)
    q = torch.zeros((17, 2, 16, 64), dtype=torch.bfloat16)
    kv = torch.zeros((64, 16, 128), dtype=torch.bfloat16)
    ids = torch.randint(-2, 80, (34, 7), generator=rng, dtype=torch.int32)
    ids[0] = torch.tensor([1, 1, 1, 2, 2, 3, 3])
    counts = torch.randint(0, 8, (34,), generator=rng, dtype=torch.int32)
    counts[0] = 7
    counts[-1] = 0
    for qt in (2, 4, 8):
        w = allocate_workspace(q, kv, ids, qt)
        _union[(len(w.lengths),)](
            ids,
            counts,
            w.union,
            w.lengths,
            17,
            2,
            7,
            64,
            qt,
            triton.next_power_of_2(qt * 7),
        )
        _membership[(len(w.lengths), qt * 7)](
            ids,
            counts,
            w.union,
            w.lengths,
            w.occurrences,
            17,
            2,
            7,
            qt,
            8,
        )
        for tile in range(len(w.lengths)):
            first, head = tile // 2 * qt, tile % 2
            rows = [
                ids[q * 2 + head, : counts[q * 2 + head]].tolist() if q < 17 else []
                for q in range(first, first + qt)
            ]
            union = sorted({x for row in rows for x in row if 0 <= x < 64})
            assert w.lengths[tile] == len(union)
            assert w.union[tile, : len(union)].tolist() == union
            for slot, physical in enumerate(union):
                assert w.occurrences[tile, slot].tolist() == [
                    row.count(physical) for row in rows
                ]


def test_gpu_grouping_in_cpu_interpreter():
    # Start before importing Triton so its sort helpers also run interpreted.
    # CPU tensor pointers are used throughout; this must not occupy the H100.
    script = f"import runpy; runpy.run_path({str(Path(__file__))!r})['_interpret_grouping_cases']()"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
    )


def test_oracle_duplicate_weight_causal_tail_and_empty_rows():
    q = torch.zeros((3, 1, 16, 64), dtype=torch.bfloat16)
    kv = torch.zeros((2, 16, 128), dtype=torch.bfloat16)
    kv[0, :, 64:] = 1
    kv[1, :, 64:] = 4
    kv[1, 8:] = torch.nan
    ids = torch.tensor([[0, 1, -1], [0, 0, 1], [0, 1, 0]], dtype=torch.int32)
    counts = torch.tensor([2, 3, 0], dtype=torch.int32)
    starts = torch.tensor([0, 16], dtype=torch.int32)
    positions = torch.tensor([23, 23, 23], dtype=torch.int32)
    output = selected_reference(q, kv, ids, starts, positions, counts)
    torch.testing.assert_close(output[0], torch.full_like(output[0], 2.0))
    torch.testing.assert_close(output[1], torch.full_like(output[1], 1.6))
    assert output[2].count_nonzero() == 0
    assert torch.isfinite(output).all()


def test_workspace_pipeline_is_opt_in_and_mode_mismatch_rejected():
    q, kv, ids, starts, positions, counts = make_case("high")
    assert allocate_workspace(q, kv, ids).pipelined is False
    workspace = allocate_workspace(q, kv, ids, pipelined=True)
    assert workspace.pipelined is True
    with pytest.raises(ValueError, match="Pipeline mode"):
        query_tile_attention(q, kv, ids, starts, positions, counts, workspace=workspace)
    with pytest.raises(TypeError, match="boolean"):
        allocate_workspace(q, kv, ids, pipelined=1)


@pytest.mark.gpu
@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.parametrize("tile_queries", [2, 4, 8])
def test_query_tile_sanitizes_tail_using_only_member_positions(tile_queries, pipelined):
    q = torch.zeros((2, 1, 16, 64), dtype=torch.bfloat16)
    kv = torch.full((1, 16, 128), torch.nan, dtype=torch.bfloat16)
    kv[0, :3] = 1
    ids = torch.tensor([[0], [-1]], dtype=torch.int32)
    starts = torch.tensor([16], dtype=torch.int32)
    positions = torch.tensor([18, 31], dtype=torch.int32)
    counts = torch.tensor([1, 0], dtype=torch.int32)
    expected = selected_reference(q, kv, ids, starts, positions, counts)
    actual = query_tile_attention(
        *[x.cuda() for x in (q, kv, ids, starts, positions, counts)],
        tile_queries=tile_queries,
        pipelined=pipelined,
    )
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.03, atol=0.03)


@pytest.mark.gpu
def test_pipeline_reused_buffers_have_independent_member_tails():
    # Five union blocks alternate early/late query ownership, reusing both
    # shared buffers. Early-only blocks contain NaN after token 2; the later
    # query must neither expose that tail nor lose its own valid tail.
    q = torch.zeros((2, 1, 16, 64), dtype=torch.bfloat16)
    kv = torch.zeros((5, 16, 128), dtype=torch.bfloat16)
    for block in range(5):
        kv[block, :, 64:] = block + 1
        if block % 2 == 0:
            kv[block, 3:] = torch.nan
    ids = torch.tensor([[0, 2, 0, 4], [1, 3, -1, -1]], dtype=torch.int32)
    starts = torch.full((5,), 16, dtype=torch.int32)
    positions = torch.tensor([18, 31], dtype=torch.int32)
    counts = torch.tensor([4, 2], dtype=torch.int32)
    expected = selected_reference(q, kv, ids, starts, positions, counts)
    actual = query_tile_attention(
        *[x.cuda() for x in (q, kv, ids, starts, positions, counts)],
        tile_queries=2,
        pipelined=True,
    )
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("overlap", ["low", "medium", "high"])
@pytest.mark.parametrize("tile_queries", [2, 4, 8])
@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.gpu
def test_query_tile_numerical_membership_duplicates_mask_and_graph(
    overlap, tile_queries, pipelined
):
    from cxl_recsys.models.nsa.backends.tilelang import build_kernel

    cpu = make_case(overlap)
    expected = selected_reference(*cpu)
    q, kv, ids, starts, positions, counts = [x.cuda() for x in cpu]
    workspace = allocate_workspace(q, kv, ids, tile_queries, pipelined=pipelined)
    actual = query_tile_attention(
        q,
        kv,
        ids,
        starts,
        positions,
        counts,
        tile_queries=tile_queries,
        workspace=workspace,
        pipelined=pipelined,
    )
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.03, atol=0.03)
    # Existing single-query consumer is the numerical and implementation control.
    flat = q.flatten(0, 1)
    state = torch.empty(flat.shape[:2], device=q.device)
    acc = torch.empty(flat.shape, device=q.device)
    single = torch.empty_like(flat)
    build_kernel(len(flat), 16, len(kv), 16, ids.shape[1])(
        flat,
        kv,
        ids,
        starts,
        positions,
        counts,
        state,
        state,
        acc,
        single,
    )
    torch.testing.assert_close(actual.flatten(0, 1), single, rtol=0.03, atol=0.03)
    # Replaying changes references, counts, Q and causal positions without
    # recapture. Padding/empty rows never acquire another query's membership.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        group_queries(ids, counts, workspace)
        consume_grouped(q, kv, starts, positions, workspace)
    counts.zero_()
    graph.replay()
    assert workspace.output.count_nonzero().item() == 0
    counts.copy_(cpu[-1])
    q.neg_()
    graph.replay()
    cpu = (-cpu[0], *cpu[1:])
    torch.testing.assert_close(
        workspace.output.cpu().double(),
        selected_reference(*cpu),
        rtol=0.03,
        atol=0.03,
    )
