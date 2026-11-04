"""Independent consumption-stream oracle and static IO-order regressions."""

import pytest
import torch


def dependency_oracle(ids, counts, flags, consumer_ctas, positions, *, shared=False):
    """Walk each CTA's serial stream; count logical rows, not physical CTAs."""
    occurrences = {block: [] for block, missing in enumerate(flags) if missing}
    for consumer in range(consumer_ctas):
        depth = 0
        for row in range(consumer, len(ids), consumer_ctas):
            if positions[row] < 0:
                continue
            for index, block in enumerate(ids[row][: counts[row]]):
                if block in occurrences:
                    occurrences[block].append((depth + index, row))
            # Resident references still consume attention steps.
            depth += counts[row]
    first = {
        block: min((depth for depth, _ in uses), default=float("inf"))
        for block, uses in occurrences.items()
    }
    near = {
        block: len({row for depth, row in uses if depth < first[block] + 4})
        for block, uses in occurrences.items()
    }
    queue = sorted(
        occurrences,
        key=lambda block: (first[block], -near[block] if shared else 0, block),
    )
    return queue, first, near


def _case():
    # Block 2 repeats within row 0, but row 1's depth-4 reference is outside
    # its near window. Block 3 has two near consumers, so it wins the tie.
    return dict(
        ids=[
            [2, 2, 2, 3, 2, 2],
            [3, 0, 0, 0, 2, 5],
            [1, 4, -1, -1, -1, -1],
            [1, 1, 1, 1, -1, -1],
            [7, 7, 7, 7, 7, 7],
            [4, -1, -1, -1, -1, -1],
        ],
        counts=[6, 6, 2, 4, 0, 1],
        flags=[0, 1, 1, 1, 1, 1, 1, 1],
        positions=[7, 7, 8, 8, -1, 9],
    )


def test_oracle_resident_steps_window_boundary_and_stable_tail():
    case = _case()
    queue, first, near = dependency_oracle(**case, consumer_ctas=2)
    assert queue == [2, 3, 5, 1, 4, 6, 7]
    assert [first[b] for b in (2, 3, 5, 1, 4)] == [0, 0, 5, 6, 7]
    assert near[2] == 1 and near[3] == 2
    assert near[4] == 2  # Depths 7 and 10, on reused consumer CTAs.
    assert near[6] == near[7] == 0
    assert dependency_oracle(**case, consumer_ctas=2, shared=True)[0] == [
        3,
        2,
        5,
        1,
        4,
        6,
        7,
    ]


def test_oracle_counts_distinct_rows_reusing_one_cta():
    queue, first, near = dependency_oracle(
        ids=[[2, 1], [1, 0], [2, -1], [-1, -1], [2, -1]],
        counts=[2, 2, 1, 0, 1],
        flags=[0, 1, 1],
        positions=[0, 0, 1, -1, 2],
        consumer_ctas=2,
        shared=True,
    )
    assert first == {1: 0, 2: 0}
    assert near == {1: 2, 2: 3}
    assert queue == [2, 1]


def _assert_queue(work, shared):
    expected, _, near = dependency_oracle(
        ids=work["all_ids"].cpu().tolist(),
        counts=work["all_counts"].cpu().tolist(),
        flags=work["flags"].cpu().tolist(),
        positions=work["positions"].cpu().tolist(),
        consumer_ctas=work["consumer_ctas"],
        shared=shared,
    )
    flags = work["flags"].cpu().bool()
    queue, ranks = work["ioids"].cpu(), work["ranks"].cpu()
    missing = int(work["missing"].item())
    assert missing == int(flags.sum()) == len(expected)
    assert queue[:missing].tolist() == expected
    assert len(set(expected)) == missing
    assert ranks[queue[:missing].long()].tolist() == list(range(missing))
    assert (queue[missing:] == -1).all()
    assert (ranks[~flags] == -1).all()
    if shared:
        actual = work["dependency_near_consumers"].cpu().tolist()
        assert {block: actual[block] for block in near} == near


@pytest.mark.gpu
@pytest.mark.parametrize("shared", [False, True])
def test_dependency_queue_graph_refreshes_ids_counts_and_residency(shared):
    from cxl_recsys.models.nsa.planning.active import _ranks
    from cxl_recsys.models.nsa.planning.read_order import (
        allocate_read_order,
        reorder_reads,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    work = dict(
        physical=8,
        rows=6,
        width=6,
        heads=2,
        per_head=4,
        consumer_ctas=2,
        scratch_bytes=0,
    )
    for name, shape in (
        ("all_ids", (6, 6)),
        ("all_counts", (6,)),
        ("positions", (6,)),
        ("flags", (8,)),
        ("ioids", (8,)),
        ("ranks", (8,)),
        ("missing", (1,)),
    ):
        work[name] = torch.empty(shape, dtype=torch.int32, device="cuda")
    order = "consumption_depth_shared" if shared else "consumption_depth"
    allocate_read_order(work, "cuda", order=order)

    def plan():
        _ranks[(1,)](
            work["flags"],
            work["ioids"],
            work["ranks"],
            work["missing"],
            8,
            8,
            num_warps=4,
        )
        reorder_reads(work, order)

    def populate(variant):
        case = _case()
        if variant == "rows":
            # Three distinct logical rows reuse CTA 0 within a four-step window.
            case["ids"][0] = [2, 3, -1, -1, -1, -1]
            case["ids"][2] = [2, -1, -1, -1, -1, -1]
            case["ids"][4] = [2, -1, -1, -1, -1, -1]
            case["counts"] = [2, 2, 1, 0, 1, 0]
            case["positions"] = [7, 7, 8, -1, 9, -1]
        elif variant == "residency":
            case["flags"] = [1, 0, 0, 1, 1, 0, 0, 0]
        elif variant == "empty":
            case["counts"] = [0] * 6
            case["flags"] = [0] * 8
            case["positions"] = [-1] * 6
        for source, destination in (
            ("ids", "all_ids"),
            ("counts", "all_counts"),
            ("flags", "flags"),
            ("positions", "positions"),
        ):
            work[destination].copy_(torch.tensor(case[source], dtype=torch.int32))
        return case

    populate("initial")
    for _ in range(3):
        plan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    for variant in ("initial", "rows", "residency", "empty", "initial"):
        case = populate(variant)
        graph.replay()
        torch.cuda.synchronize()
        _assert_queue(work, shared)
        assert work["all_ids"].cpu().tolist() == case["ids"]
        assert work["all_counts"].cpu().tolist() == case["counts"]
        assert work["flags"].cpu().tolist() == case["flags"]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "physical,rows,width,consumers",
    [(4097, 9, 8, 3), (1025, 513, 64, 8)],
    ids=["large_torch_sort", "small_int64_sort"],
)
def test_dependency_sort_boundaries_and_empty_replay(physical, rows, width, consumers):
    from cxl_recsys.models.nsa.planning.read_order import (
        allocate_read_order,
        reorder_reads,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ids = torch.zeros((rows, width), dtype=torch.int32)
    ids[0, :6] = torch.tensor([2, 2, 2, 3, 2, 2])
    ids[1, :6] = torch.tensor([3, 0, 0, 0, 2, 5])
    ids[-1, 0] = physical - 1
    counts = torch.full((rows,), width, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int32)
    counts[7], positions[7] = 0, -1
    flags = torch.ones(physical, dtype=torch.int32)
    flags[0], flags[6::11] = 0, 0
    # All other references are resident block 0. In the small-sort case the
    # last block is first read at depth 4096, so even a referenced key exceeds
    # signed int32; inactive and unreferenced keys must still follow it.
    if physical == 1025:
        _, first, _ = dependency_oracle(
            ids.tolist(),
            counts.tolist(),
            flags.tolist(),
            consumers,
            positions.tolist(),
            shared=True,
        )
        assert first[physical - 1] == 4096
    work = dict(physical=physical, rows=rows, width=width, consumer_ctas=consumers)
    for name, value in (
        ("all_ids", ids),
        ("all_counts", counts),
        ("positions", positions),
        ("flags", flags),
    ):
        work[name] = value.cuda()
    for name, size in (("ioids", physical), ("ranks", physical), ("missing", 1)):
        work[name] = torch.empty(size, dtype=torch.int32, device="cuda")
    work["missing"].fill_(int(flags.sum()))
    order = "consumption_depth_shared"
    allocate_read_order(work, "cuda", order=order)
    for _ in range(3):
        reorder_reads(work, order)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reorder_reads(work, order)
    for live_flags in (flags, torch.zeros_like(flags), flags):
        work["flags"].copy_(live_flags)
        work["missing"].fill_(int(live_flags.sum()))
        graph.replay()
        _assert_queue(work, shared=True)
        assert work["all_ids"].cpu().equal(ids)
        assert work["all_counts"].cpu().equal(counts)
        assert work["flags"].cpu().equal(live_flags)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "length,residency", [(128, "miss"), (256, "partial"), (769, "miss")]
)
def test_tma_dependency_order_preserves_strict_output_and_external_reads(
    length, residency
):
    from test_nsa_transport import setup_case

    from cxl_recsys.models.nsa.transport import GRSelectedTransport

    pool, cache, _, source_q, source_pos, source_blocks, fetches = setup_case(residency)
    index = torch.arange(length, device="cuda") % len(source_q)
    query = source_q[index].contiguous()
    positions = source_pos[index].contiguous()
    blocks = source_blocks[index].contiguous()
    blocks[: length // 2, 0] = torch.tensor([1, 0, 0, 4], device="cuda")
    positions[7] = -1
    blocks[7] = -1
    initial = {name: getattr(cache, name).clone() for name in ("tags", "locs", "data")}
    transports = []
    reference = None
    try:
        for order in ("physical", "consumption_depth", "consumption_depth_shared"):
            for name, value in initial.items():
                getattr(cache, name).copy_(value)
            transport = GRSelectedTransport(
                cache, consumer_backend="tma", read_order=order, cuda_graphs=True
            )
            transports.append(transport)
            previous = pool.read_bytes
            output, _ = transport.run(query, blocks, positions, fetches, 16)
            transport.last_done.synchronize()
            pool.flush_read_stats()
            work = next(reversed(transport.workspaces.values()))
            current = dict(
                output=output.cpu(),
                read_bytes=pool.read_bytes - previous,
                plan={
                    name: work[name].cpu()
                    for name in ("all_ids", "all_counts", "slots", "valid", "flags")
                },
            )
            assert torch.isfinite(current["output"]).all()
            assert current["read_bytes"] > 0
            if reference is None:
                reference = current
            else:
                torch.testing.assert_close(
                    current["output"], reference["output"], rtol=0.025, atol=0.004
                )
                assert current["read_bytes"] == reference["read_bytes"]
                for name, value in current["plan"].items():
                    assert torch.equal(value, reference["plan"][name])
                _assert_queue(work, order.endswith("_shared"))
            assert transport.last_stats["read_order"] == order
    finally:
        for transport in transports:
            transport.close()
        pool.close()
