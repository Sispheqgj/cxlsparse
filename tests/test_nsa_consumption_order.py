"""Consumption reordering keeps the selected multiset, sources and causal tails."""

import pytest
import torch

from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle
from cxl_recsys.models.nsa.transport import GRSelectedTransport


@pytest.mark.gpu
@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
def test_ready_order_duplicate_occurrences_partial_tail_and_graph(residency):
    from test_nsa_transport import setup_case

    pool, cache, backing, query, positions, blocks, fetches = setup_case(residency)
    # Put the causal-tail block before resident/earlier-read blocks, so the
    # strict prefix scan has genuine artificial dependencies.
    blocks = blocks[..., [3, 0, 1, 2]].contiguous()
    initial = {k: getattr(cache, k).clone() for k in ("tags", "locs", "data")}
    expected = selected_attention_oracle(
        query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
    )
    observations = {}
    executors = []
    try:
        for order in ["strict", "resident_first", "block_ready"]:
            executor = GRSelectedTransport(
                cache, consume_order=order, chunk_blocks=2, io_ctas=4, cuda_graphs=True
            )
            executors.append(executor)
            for repeat in range(2):
                for k, v in initial.items():
                    getattr(cache, k).copy_(v)
                before = pool.read_bytes
                output, counters = executor.run(query, blocks, positions, fetches, 16)
                executor.last_done.synchronize()
                pool.flush_read_stats()
                torch.testing.assert_close(
                    output.cpu().double(), expected, rtol=0.03, atol=0.03
                )
                w = next(reversed(executor.workspaces.values()))
                ids, counts, all_ids, all_counts = [
                    w[k].cpu() for k in ("ids", "counts", "all_ids", "all_counts")
                ]
                for row in range(len(all_ids)):
                    consumed = [
                        int(v)
                        for st in range(len(counts))
                        for v in ids[st, row, : int(counts[st, row])]
                    ]
                    selected = all_ids[row, : int(all_counts[row])].tolist()
                    assert sorted(consumed) == sorted(selected)
                    if order == "strict":
                        assert consumed == selected
                current = (
                    pool.read_bytes - before,
                    w["slots"].cpu(),
                    w["ioids"][: int(w["missing"].item())].cpu(),
                    w["missing"].cpu(),
                    output.cpu(),
                )
                if order == "strict":
                    observations[repeat] = current
                else:
                    ref = observations[repeat]
                    assert current[0] == ref[0]
                    for i in [1, 2, 3]:
                        assert torch.equal(current[i], ref[i])
                    torch.testing.assert_close(current[4], ref[4], rtol=0.03, atol=0.03)
                assert executor.last_stats["consume_order"] == order
    finally:
        for e in executors:
            e.close()
        pool.close()
