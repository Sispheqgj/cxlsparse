"""Real mapped-host sources, mixed tiers and bounded TMA consumer reuse."""

import pytest
import torch

from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle
from cxl_recsys.models.nsa.transport import GRSelectedTransport


@pytest.mark.gpu
@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
def test_tma_live_sources_long_queries_and_replay(residency):
    from test_nsa_transport import setup_case

    pool, cache, backing, source_q, source_pos, source_blocks, fetches = setup_case(
        residency
    )
    transport = GRSelectedTransport(cache, consumer_backend="tma", capacity=3)
    try:
        for length in (13, 129, 769):
            index = torch.arange(length, device="cuda") % len(source_q)
            query = source_q[index].contiguous()
            positions = source_pos[index].contiguous()
            blocks = source_blocks[index].contiguous()
            for replay in range(2):
                if replay:
                    query.mul_(-0.75)
                    backing[67:].add_(0.25)
                    for head, fetch in enumerate(fetches):
                        fetch["current_kv"].copy_(backing[67:, head])
                expected = selected_attention_oracle(
                    query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
                )
                previous = pool.read_bytes
                output, deferred = transport.run(query, blocks, positions, fetches, 16)
                transport.last_done.synchronize()
                counts = deferred.materialize(deferred.statistics.tolist())
                pool.flush_read_stats()
                torch.testing.assert_close(
                    output.cpu().double(), expected, rtol=0.03, atol=0.03
                )
                assert (
                    pool.read_bytes - previous == sum(c["misses"] for c in counts) * 256
                )
                stats = transport.last_stats
                assert stats["consumer"] == "tma_bounded_rows"
                assert stats["consumer_ctas"] <= 256
                assert stats["effective_q"] == length
                assert stats["cpu_selection_downloads"] == 0
                if length == 13 and replay == 0:
                    work = next(reversed(transport.workspaces.values()))
                    kinds = work["tma_source_kinds"].cpu().tolist()
                    if residency == "miss":
                        assert 1 in kinds and 2 in kinds
                    elif residency == "partial":
                        assert 2 in kinds
                    else:
                        assert set(kinds) == {0}
                captures = transport.graphs.captures
                transport.run(query, blocks, positions, fetches, 16)
                transport.last_done.synchronize()
                assert transport.graphs.captures == captures
    finally:
        transport.close()
        pool.close()


@pytest.mark.gpu
@pytest.mark.parametrize("window", [1, 2, 4])
def test_tma_preparation_poisoned_graph_and_hot_commit(window, monkeypatch):
    from test_nsa_transport import setup_case

    from cxl_recsys.models.nsa.tma import transport as nsa_tma_transport

    pool, cache, backing, query, positions, blocks, fetches = setup_case("partial")
    # Unrequested suffix values must not leak through a partial source block or
    # into its hot commit. Earlier query rows also mask finite future tokens.
    backing[73:] = torch.nan
    positions.clamp_max_(72)
    original_execute = nsa_tma_transport.execute

    def poison_and_execute(q, work):
        work["kv"].fill_(torch.nan)
        work["tma_ready"].fill_(73)
        return original_execute(q, work)

    monkeypatch.setattr(nsa_tma_transport, "execute", poison_and_execute)
    transport = GRSelectedTransport(
        cache,
        consumer_backend="tma",
        read_order="consumption_depth",
        ready_window=window,
        cuda_graphs=True,
    )
    try:
        captures = None
        for state in ("partial", "miss", "hit", "empty"):
            query.mul_(-0.75)
            backing[67:73].add_(0.25)
            for head, fetch in enumerate(fetches):
                fetch["current_kv"].copy_(backing[67:, head])
            if state != "partial":
                cache.tags.fill_(-1)
            if state == "miss":
                blocks[:, 0] = torch.tensor([2, 4, 2, -1], device="cuda")
                blocks[:, 1] = torch.tensor([1, 0, 4, -1], device="cuda")
            elif state == "hit":
                for head in range(2):
                    cache.tags[head, :67] = torch.arange(67, device="cuda")
                    cache.data[cache.locs[head, :67].long()] = backing[:67, head].cuda()
            elif state == "empty":
                blocks.fill_(-1)
            blocks[3] = -1
            positions[7], blocks[7] = -1, -1
            hot = [set(row.tolist()) - {-1} for row in cache.tags.cpu()]
            cpu_blocks, cpu_positions = blocks.cpu(), positions.cpu()
            needed = [set(), set()]
            for selected, position in zip(
                cpu_blocks.tolist(), cpu_positions.tolist(), strict=True
            ):
                for head, row in enumerate(selected):
                    for block in row:
                        if block >= 0:
                            needed[head].update(
                                range(
                                    block * 16, min(block * 16 + 16, position + 1, 67)
                                )
                            )
            expected_bytes = (
                sum(len(tokens - tags) for tokens, tags in zip(needed, hot)) * 256
            )
            expected = selected_attention_oracle(
                query.cpu(), backing, cpu_blocks, cpu_positions, 16
            )
            before = pool.read_bytes
            output, _ = transport.run(query, blocks, positions, fetches, 16)
            transport.last_done.synchronize()
            pool.flush_read_stats()
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=0.03, atol=0.03
            )
            assert pool.read_bytes - before == expected_bytes
            work = next(reversed(transport.workspaces.values()))
            assert torch.all(work["tma_ready"] == 1)
            # Ready alone cannot certify payload initialization: check every
            # actually committed token against its immutable historical value.
            for head in range(2):
                tags = cache.tags[head].cpu()
                occupied = tags >= 0
                assert set(tags[occupied].tolist()) == needed[head]
                locations = cache.locs[head].cpu()[occupied].long()
                actual = cache.data[locations.cuda()].cpu()
                assert torch.equal(
                    actual.view(torch.uint8),
                    backing[tags[occupied].long(), head].contiguous().view(torch.uint8),
                )
            if state == "empty":
                assert torch.isnan(work["kv"]).all()
            if captures is None:
                captures = transport.graphs.captures
            else:
                assert transport.graphs.captures == captures
    finally:
        transport.close()
        pool.close()
