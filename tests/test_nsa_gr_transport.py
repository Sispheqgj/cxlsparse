"""GR routing budgets and actual compact transport/consumer composition."""

from types import SimpleNamespace

import pytest
import torch

from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle
from cxl_recsys.models.nsa.transport import GRSelectedTransport, workspace_estimate


def test_budget_route_precedes_any_gpu_or_pool_access():
    executor = GRSelectedTransport.__new__(GRSelectedTransport)
    executor.pool = SimpleNamespace(backend="cxl")
    executor.read_order = "physical"
    executor.consumer_backend = "legacy"
    executor.cross_query_dedup = True
    executor.chunk_blocks, executor.query_group = 64, 1
    executor.workspace_budget = 1
    assert executor.refusal(105, 8192, 2, 16, 16, 16) == "workspace_budget"
    assert executor.refusal(1025, 8192, 2, 16, 16, 16) == "query_bucket_limit"
    executor.workspace_budget = 512 * 1024**2
    for h in (4096, 8192, 16384, 32768, 65536):
        for q in (63, 64, 65, 105, 127, 128, 129, 255, 256, 257, 512, 1024):
            assert executor.refusal(q, h, 2, 16, 16, 16) is None
    assert workspace_estimate(65, 8215, 2, 16, 16, 16) == workspace_estimate(
        127, 8215, 2, 16, 16, 16
    )
    small_chunks = workspace_estimate(128, 65536, 2, 16, 16, 16, chunk_blocks=1)
    regular = workspace_estimate(128, 65536, 2, 16, 16, 16)
    assert small_chunks > regular
    executor.workspace_budget = regular
    executor.chunk_blocks = 1
    assert executor.refusal(128, 65536, 2, 16, 16, 16) == "workspace_budget"


@pytest.mark.gpu
@pytest.mark.parametrize("group", [1, 2, 4, 8])
def test_gr_shared_union_replay_padding_hot_and_io(group):
    from test_nsa_transport import setup_case

    pool, cache, backing, source_q, source_pos, source_blocks, fetches = setup_case(
        "partial"
    )
    executor = GRSelectedTransport(cache, query_group=group, capacity=5, io_ctas=4)
    captures = {}
    try:
        for q in (65, 105, 127, 128, 63, 64, 129, 255, 257, 512, 1024):
            index = torch.arange(q, device="cuda") % len(source_q)
            query, positions, blocks = (
                source_q[index],
                source_pos[index],
                source_blocks[index],
            )
            expected = selected_attention_oracle(
                query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
            )
            previous = pool.read_bytes
            output, counters = executor.run(query, blocks, positions, fetches, 16)
            executor.last_done.synchronize()
            counts = counters.materialize(counters.statistics.tolist())
            pool.flush_read_stats()
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=0.03, atol=0.03
            )
            assert (
                pool.read_bytes - previous == sum(row["misses"] for row in counts) * 256
            )
            assert all(
                row["old_references"] + row["local_tokens"] > 0 for row in counts
            )
            stats = executor.last_stats
            assert stats["effective_q"] == q and stats["cpu_selection_downloads"] == 0
            assert stats["consumer_groups"] <= 8
            key = stats["query_bucket"]
            if key in captures:
                assert executor.graphs.captures == captures[key]
            else:
                captures[key] = executor.graphs.captures
            for head in range(2):
                ids = blocks[:, head].cpu()[..., None] * 16 + torch.arange(16)
                allowed = (
                    (blocks[:, head].cpu()[..., None] >= 0)
                    & (ids <= positions.cpu()[:, None, None])
                    & (ids < 67)
                )
                frequency = torch.bincount(ids[allowed], minlength=256)
                ordered = torch.argsort(frequency, descending=True, stable=True)
                ordered = ordered[frequency[ordered] > 0][: cache.hot]
                tags = cache.tags[head, : cache.hot].cpu()
                assert tags[tags >= 0].tolist() == ordered.tolist()
    finally:
        executor.close()
        pool.close()


def test_private_budget_tracks_query_occurrences_instead_of_history_union():
    options = dict(all_hbm=True, native=True, cross_query_dedup=False)
    short_history = workspace_estimate(256, 4096, 2, 16, 16, 16, **options)
    shared = workspace_estimate(256, 4096, 2, 16, 16, 16, all_hbm=True, native=True)
    assert short_history > shared
    assert short_history < 512 * 1024**2
