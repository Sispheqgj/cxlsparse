"""Consumption-row semantics and the TMA planner's scratch footprint."""

from types import SimpleNamespace

import pytest
import torch

from cxl_recsys.models.nsa.planning.active import ActivePlanAdapter
from cxl_recsys.models.nsa.planning.occurrences import build_occurrences


def test_occurrence_only_allocation_omits_unused_stage_storage():
    adapter = ActivePlanAdapter(
        SimpleNamespace(
            hot=256,
            pool=SimpleNamespace(backend="cxl", region_count=2),
        )
    )
    query = torch.empty((128, 2, 16, 64), device="meta")
    blocks = torch.empty((128, 2, 16), device="meta", dtype=torch.int32)
    fetch = dict(cached_length=8192, current_kv=range(128))
    legacy = adapter.allocate(query, blocks, [fetch] * 2, 16, all_hbm=True)
    compact = adapter.allocate(
        query, blocks, [fetch] * 2, 16, all_hbm=True, occurrence_only=True
    )

    omitted = ("ids", "counts", "barriers", "compute_trace", "io_trace")
    assert legacy["occurrence_only"] is False
    assert compact["occurrence_only"] is True
    assert compact["stages"] == legacy["stages"] == 2
    for name in omitted:
        assert legacy[name].numel() > 0
        assert compact[name].numel() == 0
    for name, value in legacy.items():
        if isinstance(value, torch.Tensor) and name not in omitted:
            assert compact[name].shape == value.shape
            assert compact[name].dtype == value.dtype
    saved = sum(legacy[name].numel() * legacy[name].element_size() for name in omitted)
    assert legacy["scratch_bytes"] - compact["scratch_bytes"] == saved
    assert legacy["reset_bytes"] - compact["reset_bytes"] == legacy["ids"].numel() * 4


@pytest.mark.gpu
@pytest.mark.parametrize("error_head", [-1, 0, 1])
def test_occurrences_keep_duplicates_order_and_exclude_invalid_rows(error_head):
    blocks = torch.tensor(
        [
            [[1, 0, 0, 2, -1, 6, 3], [1, 0, 1, -1, 2, 6, 0]],
            [[4, 2, 2, 3, 0, 5, -1], [4, 3, 2, 1, 0, -1, 6]],
            [[0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 0]],
        ],
        dtype=torch.int32,
    )
    positions = torch.tensor([31, 79, -1], dtype=torch.int32)
    mapping = torch.tensor(
        [[2, -1, 0, 4, -2, 3, -2], [1, 0, 3, 2, 4, -1, -2]],
        dtype=torch.int32,
    )
    stats = torch.zeros((2, 7), dtype=torch.int32)
    if error_head >= 0:
        stats[error_head, 6] = 1
    expected = torch.full((6, 7), -1, dtype=torch.int32)
    expected_counts = torch.zeros(6, dtype=torch.int32)
    for query in range(3):
        for head in range(2):
            selected = []
            if positions[query] >= 0 and head != error_head:
                for block in blocks[query, head].tolist():
                    if 0 <= block < 6 and block * 16 <= positions[query]:
                        compact = mapping[head, block].item()
                        if 0 <= compact < 4:
                            selected.append(head * 4 + compact)
            row = query * 2 + head
            expected[row, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
            expected_counts[row] = len(selected)

    work = dict(
        rows=6,
        heads=2,
        width=7,
        map_blocks=6,
        per_head=4,
        bs=16,
        mapping=mapping.cuda(),
        stats=stats.cuda(),
        all_ids=torch.full((6, 7), -1, dtype=torch.int32, device="cuda"),
        all_counts=torch.empty(6, dtype=torch.int32, device="cuda"),
        positions=torch.empty(6, dtype=torch.int32, device="cuda"),
    )
    build_occurrences(blocks.cuda(), positions.cuda(), work)
    torch.testing.assert_close(work["all_ids"].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(work["all_counts"].cpu(), expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(
        work["positions"].cpu(), positions.repeat_interleave(2), rtol=0, atol=0
    )


@pytest.mark.gpu
def test_prepared_ready_keeps_cooperative_launch_without_grid_barrier(monkeypatch):
    from cxl_recsys.models.nsa.tma.pipeline import build_tma_kernel

    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "cython")
    for prepared in (True, False):
        kernel = build_tma_kernel.__wrapped__(
            4,
            8,
            width=4,
            route="staged",
            io_ctas=2,
            consumer_ctas=2,
            copy_kind="bulk",
            resident_ready=True,
            ready_prepared=prepared,
        )
        assert "cudaLaunchCooperativeKernel" in kernel.get_host_source()
        assert ("this_grid().sync()" in kernel.get_kernel_source()) is not prepared
