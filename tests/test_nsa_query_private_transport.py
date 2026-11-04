"""Private payloads through batch/native and staged TMA consumers."""

import pytest
import torch
from test_nsa_query_private_plan import _assert_private_plan, _descriptors

from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle
from cxl_recsys.models.nsa.transport import GRSelectedTransport


@pytest.mark.gpu
@pytest.mark.parametrize("consumer", ["native", "tma"])
@pytest.mark.parametrize("length", [128, 256])
def test_private_consumers_replay_independent_reads_and_one_hot_policy(consumer, length):
    from test_nsa_transport import setup_case

    pool, cache, backing, source_q, source_pos, source_blocks, fetches = setup_case("miss")
    index = torch.arange(length, device="cuda") % len(source_q)
    query, positions, blocks = (source[index].contiguous() for source in
                                (source_q, source_pos, source_blocks))
    transport = GRSelectedTransport(
        cache, consumer_backend=consumer, cross_query_dedup=False,
        read_order="consumption_depth" if consumer == "tma" else "physical",
        cuda_graphs=True,
    )
    captures = None
    try:
        for state in ("miss", "hit", "changed_miss", "empty"):
            if state in ("miss", "changed_miss", "empty"):
                cache.tags.fill_(-1)
            if state == "changed_miss":
                query.mul_(-0.75)
                backing[67:].add_(0.25)
                for head, fetch in enumerate(fetches):
                    fetch["current_kv"].copy_(backing[67:, head])
                blocks[0, :, 0] = 2
                positions[1], blocks[1] = -1, -1
            elif state == "empty":
                blocks.fill_(-1)
            _, references = _descriptors(cache, fetches)
            expected = selected_attention_oracle(
                query.cpu(), backing, blocks.cpu(), positions.cpu(), 16
            )
            before = pool.read_bytes
            output, deferred = transport.run(query, blocks, positions, fetches, 16)
            transport.last_done.synchronize()
            counts = deferred.materialize(deferred.statistics.tolist())
            pool.flush_read_stats()
            torch.testing.assert_close(output.cpu().double(), expected,
                                       rtol=0.03, atol=0.03)
            work = next(reversed(transport.workspaces.values()))
            shared_reference = _assert_private_plan(
                work, blocks, positions, references, cache.hot
            )
            sources = work["slots"].cpu()
            external = sources[sources >= 0]
            assert pool.read_bytes - before == len(external) * 256
            assert len(external) == sum(row["misses"] for row in counts)
            if state in ("miss", "changed_miss"):
                assert len(external) > len(external.unique())
                assert len(external) > sum(row[3] for row in shared_reference.stats)
            else:
                assert len(external) == 0
            # Hot decisions use all queries once; private destinations must not
            # change the frequency ordering or commit a stale payload copy.
            for head in range(2):
                wanted = list(shared_reference.hot[head])
                tags = cache.tags[head, :cache.hot].cpu()
                assert tags[tags >= 0].tolist() == wanted
                actual = cache.data[cache.locs[head, :len(wanted)].long()].cpu()
                torch.testing.assert_close(actual, backing[wanted, head],
                                           rtol=0, atol=0)
            stats = transport.last_stats
            assert stats["planner"] == "query_private_gpu"
            assert stats["cross_query_dedup"] is False
            assert stats["active_block_capacity"] == length * 2 * blocks.shape[-1]
            assert stats["consumer"] == (
                "tma_bounded_rows" if consumer == "tma" else "native_indexed_bounded"
            )
            if captures is None:
                captures = transport.graphs.captures
            else:
                assert transport.graphs.captures == captures
    finally:
        transport.close()
        pool.close()
