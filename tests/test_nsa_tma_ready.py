"""Bounded ready selection and live mapped-host attention under Graph replay."""

from pathlib import Path

import pytest
import torch

from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("window", [1, 2, 4])
def test_ready_selector_original_window_and_occurrences(window):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 required")

    import tilelang
    from tilelang import language as T

    from cxl_recsys.models.nsa.backends.tilelang import pass_configs
    from cxl_recsys.models.nsa.tma import pipeline as nsa_tma_pipeline

    # Each expected tuple is for Window=1/2/4. Holes never enlarge the original
    # index window; identical physical IDs retain distinct occurrence bits.
    cases = [
        (list(range(32)), [1, 3], 0b1111, (-1, 1, 1)),
        (list(range(32)), [3], 0b1101, (-1, -1, 3)),
        (list(range(32)), [4, 5], 0b110010, (-1, -1, 4)),
        (list(range(32)), [5], 0b110010, (-1, -1, -1)),
        (list(range(32)), [1, 2], 0b1101, (-1, -1, 2)),
        ([3, 3, 1, 3] + [0] * 28, [3], 0b1111, (0, 0, 0)),
        ([3, 3, 1, 3] + [0] * 28, [3], 0b1110, (1, 1, 1)),
        ([3, 3, 1, 3] + [0] * 28, [3], 0b1100, (-1, 3, 3)),
        ([3, 3, 1, 3] + [0] * 28, [3], 0b0100, (-1, -1, -1)),
        ([7, 3, 2, 0] + [0] * 28, [2, 3], 0b1111, (-1, 1, 1)),
        (list(range(32)), [31], 1 << 31, (31, 31, 31)),
        (list(range(32)), list(range(32)), 0, (-1, -1, -1)),
        (list(range(32)), [], 0b1111, (-1, -1, -1)),
    ]
    count = len(cases)
    directory = Path(nsa_tma_pipeline.__file__).parent
    source = "\n".join(
        path.read_text()
        for path in (
            directory / "pipeline.cuh",
            directory.parent / "experimental/ready.cuh",
        )
    )

    @T.prim_func
    def select(
        IDs: T.Tensor((count, 32), "int32"),
        Flags: T.Tensor((count, 32), "int32"),
        Remaining: T.Tensor((count,), "uint32"),
        Out: T.Tensor((count, 32), "int32"),
    ):
        with T.Kernel(count, threads=32) as row:
            T.import_source(source)
            lane = T.get_thread_binding()
            Out[row, lane] = T.call_extern(
                "int32",
                f"nsa_pick_ready<{window}>",
                T.address_of(IDs[row, 0]),
                T.address_of(Flags[row, 0]),
                Remaining[row],
            )

    kernel = tilelang.compile(
        select,
        target={"kind": "cuda", "arch": "sm_90a"},
        pass_configs=pass_configs(),
    )
    ids = torch.tensor([case[0] for case in cases], dtype=torch.int32, device="cuda")
    flags = torch.tensor(
        [[int(i in case[1]) for i in range(32)] for case in cases],
        dtype=torch.int32,
        device="cuda",
    )
    remaining = torch.tensor(
        [case[2] for case in cases], dtype=torch.uint32, device="cuda"
    )
    actual = torch.empty((count, 32), dtype=torch.int32, device="cuda")
    kernel(ids, flags, remaining, actual)
    column = (1, 2, 4).index(window)
    expected = torch.tensor([case[3][column] for case in cases], dtype=torch.int32)
    assert torch.equal(actual.cpu(), expected[:, None].expand(-1, 32))


def _expected_plan(blocks, positions, hot, work):
    """CPU set accounting, independent of the GPU compact plan and read queue."""
    heads = blocks.shape[1]
    references, needed = [], [dict() for _ in range(heads)]
    for selections, position in zip(blocks.tolist(), positions.tolist(), strict=True):
        for head, selected in enumerate(selections):
            row = [block for block in selected if block >= 0 and block * 16 <= position]
            references.append(row)
            for block in row:
                needed[head].setdefault(block, set()).update(
                    range(block * 16, min(block * 16 + 16, position + 1, 80))
                )
    ids = torch.full(work["all_ids"].shape, -1, dtype=torch.int32)
    counts = torch.zeros(work["all_counts"].shape, dtype=torch.int32)
    flags = torch.zeros(work["flags"].shape, dtype=torch.int32)
    mappings, misses = [], 0
    for head, by_block in enumerate(needed):
        mapping = {
            block: head * work["per_head"] + index
            for index, block in enumerate(sorted(by_block))
        }
        mappings.append(mapping)
        for block, tokens in by_block.items():
            external = {token for token in tokens if token < 67} - hot[head]
            flags[mapping[block]] = bool(external)
            misses += len(external)
    for row, selected in enumerate(references):
        counts[row] = len(selected)
        ids[row, : len(selected)] = torch.tensor(
            [mappings[row % heads][block] for block in selected], dtype=torch.int32
        )
    return ids, counts, flags, misses * 256


@pytest.mark.parametrize("window", [2, 4])
@pytest.mark.parametrize(
    "length,residency", [(13, "partial"), (129, "hit"), (769, "miss")]
)
def test_ready_window_mapped_sources_oracle_and_graph_mutation(
    window, length, residency
):
    from test_nsa_transport import setup_case

    from cxl_recsys.models.nsa.transport import GRSelectedTransport

    pool, cache, backing, source_q, source_pos, source_blocks, fetches = setup_case(
        residency
    )
    index = torch.arange(length, device="cuda") % len(source_q)
    query = source_q[index].contiguous()
    positions = source_pos[index].contiguous()
    blocks = source_blocks[index].contiguous()
    blocks[3] = -1
    positions[7], blocks[7] = -1, -1
    transport = GRSelectedTransport(
        cache,
        consumer_backend="tma",
        read_order="consumption_depth",
        ready_window=window,
        cuda_graphs=True,
    )
    try:
        captures = None
        for replay in range(2):
            if replay:
                # Change source values, physical selections, prefix Counts and
                # missing flags without changing geometry or Graph addresses.
                query.mul_(-0.75)
                backing[67:].add_(0.25)
                cache.tags.fill_(-1)
                hot = [0, 1, 2, 17, 65] if residency == "miss" else []
                for head, fetch in enumerate(fetches):
                    fetch["current_kv"].copy_(backing[67:, head])
                    if hot:
                        cache.tags[head, : len(hot)] = torch.tensor(
                            hot, dtype=torch.int32, device="cuda"
                        )
                        cache.data[cache.locs[head, : len(hot)].long()] = backing[
                            hot, head
                        ].cuda()
                blocks[:, 0] = torch.tensor([4, -1, 2, 2], device="cuda")
                blocks[:, 1] = torch.tensor([2, 4, 0, -1], device="cuda")
                blocks[::5] = -1
                positions.copy_(source_pos.flip(0)[index])
                positions[7], blocks[7] = -1, -1
            hot = [set(tags.tolist()) - {-1} for tags in cache.tags.cpu()]
            cpu_blocks, cpu_positions = blocks.cpu(), positions.cpu()
            expected = selected_attention_oracle(
                query.cpu(), backing, cpu_blocks, cpu_positions, 16
            )
            before = pool.read_bytes
            output, deferred = transport.run(query, blocks, positions, fetches, 16)
            transport.last_done.synchronize()
            pool.flush_read_stats()
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=0.03, atol=0.03
            )
            work = next(reversed(transport.workspaces.values()))
            ids, counts, flags, read_bytes = _expected_plan(
                cpu_blocks, cpu_positions, hot, work
            )
            assert torch.equal(work["all_ids"].cpu(), ids)
            assert torch.equal(work["all_counts"].cpu(), counts)
            assert torch.equal(work["flags"].cpu(), flags)
            assert int(work["missing"].item()) == int(flags.sum())
            assert pool.read_bytes - before == read_bytes
            counters = deferred.materialize(deferred.statistics.tolist())
            assert sum(counter["misses"] for counter in counters) * 256 == read_bytes
            assert transport.last_stats["ready_window"] == window
            assert transport.last_stats["read_order"] == "consumption_depth"
            assert transport.last_stats["consumer_ctas"] <= 256
            if captures is None:
                captures = transport.graphs.captures
            else:
                assert transport.graphs.captures == captures
    finally:
        transport.close()
        pool.close()
