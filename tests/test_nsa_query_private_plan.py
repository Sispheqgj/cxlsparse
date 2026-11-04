"""Query-private transfers: separate misses, unchanged hot selection and masks."""

from types import SimpleNamespace

import pytest
import torch

from cxl_recsys.models.nsa.planning.active import _copy_hbm, reference_active_plan
from cxl_recsys.models.nsa.planning.query_private import QueryPrivatePlanAdapter


@pytest.mark.parametrize("queries", [128, 256])
@pytest.mark.parametrize("backend", ["cxl", "rdma"])
def test_private_capacity_is_not_limited_by_history_union(queries, backend):
    query = torch.empty((queries, 2, 16, 64), device="meta")
    blocks = torch.empty((queries, 2, 16), device="meta", dtype=torch.int32)
    fetch = dict(cached_length=4096, current_kv=range(queries))
    adapter = QueryPrivatePlanAdapter(SimpleNamespace(
        hot=256, pool=SimpleNamespace(backend=backend)))
    w = adapter.geometry(query, blocks, [fetch, fetch], 16, all_hbm=True)
    assert w["per_head"] == queries * 16
    assert w["physical"] == queries * 16 * 2
    assert w["stages"] == 2
    assert w["sort_extent"] == 4096 + queries


def _descriptors(cache, fetches):
    desc = torch.tensor([
        (f["row"], f["cached_length"], f["history_length"],
         f["current_kv"].data_ptr(), len(f["current_kv"])) for f in fetches
    ], device="cuda", dtype=torch.int64)
    references = []
    for f in fetches:
        row = f["row"]
        tags, locations = cache.tags[row].cpu().tolist(), cache.locs[row].cpu().tolist()
        references.append(dict(
            cached_length=f["cached_length"], history_length=f["history_length"],
            total_length=f["cached_length"] + len(f["current_kv"]),
            hot_locations={t: loc for t, loc in zip(tags, locations) if t >= 0},
            host_slots=dict(enumerate(cache.host_locs[row].cpu().tolist())),
        ))
    return desc, references


def _assert_private_plan(w, blocks, positions, references, hot):
    blocks, positions = blocks.cpu(), positions.cpu()
    per_query = [reference_active_plan(
        blocks[q:q + 1], positions[q:q + 1], references,
        block_size=16, hot=hot) for q in range(len(blocks))]
    expected_stats = torch.tensor([p.stats for p in per_query]).sum(dim=0)
    assert w["stats"].cpu().tolist() == expected_stats.tolist()
    assert w["active_counts"].sum().item() == sum(len(p.keys) for p in per_query)
    all_ids, counts = w["all_ids"].cpu(), w["all_counts"].cpu()
    seen = set()
    for query, plan in enumerate(per_query):
        for head, row_keys in enumerate(plan.row_keys):
            row = query * w["heads"] + head
            ids = all_ids[row, :counts[row]].tolist()
            assert len(ids) == len(row_keys)
            own = set(ids)
            assert not (own & seen), "Queries must have distinct physical slots"
            seen |= own
            by_key = dict(zip(plan.keys, plan.required))
            sources = dict(zip(plan.keys, plan.sources))
            physical_by_key = {}
            for physical, key in zip(ids, row_keys):
                if key in physical_by_key:
                    assert physical_by_key[key] == physical
                physical_by_key[key] = physical
                assert w["starts"][physical].item() == key[1] * 16
                assert w["valid"][physical].item() == by_key[key]
                for offset, (kind, source) in enumerate(sources[key]):
                    assert w["slots"].view(-1, 16)[physical, offset].item() == (
                        source if kind == "external" else -1)
    return reference_active_plan(blocks, positions, references, block_size=16, hot=hot)


@pytest.mark.gpu
@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
@pytest.mark.parametrize("occurrence_only", [False, True])
def test_private_plan_keeps_duplicates_tails_hot_policy_and_graph_replay(
    residency, occurrence_only,
):
    from test_nsa_transport import setup_case

    pool, cache, backing, query, positions, blocks, fetches = setup_case(residency)
    adapter = QueryPrivatePlanAdapter(cache, chunk_blocks=3)
    w = adapter.allocate(query, blocks, fetches, 16, map_blocks=64,
                         occurrence_only=occurrence_only, read_order="consumption_depth")
    w["consumer_ctas"] = min(256, w["rows"])
    bases = torch.tensor([pool._cuda_mapping(query.device).address],
                         device=query.device, dtype=torch.int64)
    try:
        desc, references = _descriptors(cache, fetches)

        def plan():
            return adapter.plan(blocks, positions, desc, bases, w,
                                read_order="consumption_depth")

        chosen = plan()
        expected = _assert_private_plan(w, blocks, positions, references, cache.hot)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            chosen = plan()
        for replay in range(2):
            if replay:
                # Replay changes both occurrence identities and causal validity.
                # The former query-0 first block must not retain a stale mapping.
                blocks[0, :, 0] = 2
                positions[1] = -1
            graph.replay()
            expected = _assert_private_plan(w, blocks, positions, references, cache.hot)
            _copy_hbm[(w["physical"] * 16,)](
                w["address"], w["kv"], w["physical"] * 16)
            # Inspect each independently fetched destination, including the tail.
            for head in range(w["heads"]):
                count = w["active_counts"][head].item()
                for compact in range(count):
                    physical = head * w["per_head"] + compact
                    start, size = w["starts"][physical].item(), w["valid"][physical].item()
                    torch.testing.assert_close(w["kv"][physical, :size].cpu(),
                                               backing[start:start + size, head],
                                               rtol=0, atol=0)
        # The caller commits once for the whole layer, after every payload read.
        adapter.retain_hot(desc, chosen, w)
        for head in range(w["heads"]):
            tags = cache.tags[head, :cache.hot].cpu()
            assert tags[tags >= 0].tolist() == list(expected.hot[head])
            n = len(expected.hot[head])
            torch.testing.assert_close(cache.data[cache.locs[head, :n].long()].cpu(),
                                       backing[list(expected.hot[head]), head], rtol=0, atol=0)
    finally:
        pool.close()


@pytest.mark.gpu
def test_rdma_metadata_repeats_sources_with_independent_destinations():
    from test_nsa_transport import setup_case

    pool, cache, _, query, positions, blocks, fetches = setup_case("partial")
    backend = pool.backend
    pool.backend = "rdma"  # Metadata-only test; it does not impersonate RDMA IO.
    try:
        adapter = QueryPrivatePlanAdapter(cache)
        w = adapter.allocate(query, blocks, fetches, 16, all_hbm=True)
        desc, references = _descriptors(cache, fetches)
        bases = torch.zeros(1, device="cuda", dtype=torch.int64)
        adapter.plan(blocks, positions, desc, bases, w)
        _assert_private_plan(w, blocks, positions, references, cache.hot)
        meta = w["meta"].cpu()
        n = int(meta[0, 0]) * 16
        sources, destinations = meta[0, 1:1 + n], meta[1, 1:1 + n]
        mask = sources >= 0
        sources, destinations = sources[mask], destinations[mask]
        assert len(sources) == int(w["stats"][:, 3].sum())
        assert len(sources.unique()) < len(sources)
        assert len(destinations.unique()) == len(destinations)
        assert torch.equal(w["slots"].cpu()[destinations], sources)
    finally:
        pool.backend = backend
        pool.close()
