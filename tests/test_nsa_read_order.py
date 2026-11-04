"""Queue identity, valid-token grouping and production cooperative replay."""

from collections import Counter

import pytest
import torch

from cxl_recsys.models.nsa.transport import GRSelectedTransport


def oracle(w, order, group_size):
    flags = w["flags"].cpu().tolist()
    ids, counts, positions = [
        w[k].cpu() for k in ("all_ids", "all_counts", "positions")
    ]
    heads = w["heads"]
    first = {}
    token = -1
    for row in range(w["rows"]):
        if row % heads == 0 and positions[row] >= 0:
            token += 1
        if positions[row] < 0:
            assert counts[row] == 0
            continue
        for b in ids[row, : int(counts[row])].tolist():
            if flags[b]:
                first[b] = min(first.get(b, 2**30), token // group_size)
    physical = [i for i, v in enumerate(flags) if v]
    assert set(physical) == set(first)
    if order == "physical":
        return physical
    if order == "query_first_use":
        return sorted(physical, key=lambda b: (first[b], b))
    heads_of = w["read_head_ids"].cpu().tolist()
    counters = Counter()
    keys = {}
    for b in physical:
        h = heads_of[b]
        keys[b] = (counters[h], h)
        counters[h] += 1
    return sorted(physical, key=keys.get)


@pytest.mark.gpu
@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
def test_queue_modes_production_graph_and_changed_priorities(residency):
    from test_nsa_transport import setup_case

    from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle

    pool, cache, backing, q, pos, blocks, fetches = setup_case(residency)
    # 35 tokens, three 16-token groups; head 1 repeats one block. The middle
    # group adds no blocks; the last group's different selections test replay.
    index = torch.arange(35, device="cuda") % len(q)
    q, pos, blocks = q[index], pos[index], blocks[index]
    if residency != "empty":
        blocks[:, 1] = torch.tensor([0, 0, 0, 4], device="cuda", dtype=torch.int32)
        blocks[32:, 0] = torch.tensor([2, 1, 2, 4], device="cuda", dtype=torch.int32)
    # Interior invalid tokens must not advance supply group numbers.
    pos[7] = -1
    blocks[7] = -1
    initial = {k: getattr(cache, k).clone() for k in ("tags", "locs", "data")}
    executors = []
    ref = {}
    try:
        for order in ["physical", "head_interleaved", "query_first_use"]:
            e = GRSelectedTransport(
                cache,
                consume_order="block_ready",
                read_order=order,
                read_group_size=16,
                chunk_blocks=2,
                io_ctas=4,
                cuda_graphs=True,
            )
            executors.append(e)
            for repeat in range(3):
                for k, v in initial.items():
                    getattr(cache, k).copy_(v)
                # Mutate selections/initial residency with the same captured
                # graph shape, so stale first-use priorities cannot pass.
                bb = blocks.roll(repeat * 11, dims=0).contiguous()
                pp = pos.roll(repeat * 11).contiguous()
                if repeat == 2:
                    cache.tags[0, :16] = torch.arange(
                        16, device="cuda", dtype=torch.int32
                    )
                    cache.data[cache.locs[0, :16].long()] = backing[:16, 0].cuda()
                before = pool.read_bytes
                output, _ = e.run(q, bb, pp, fetches, 16)
                e.last_done.synchronize()
                pool.flush_read_stats()
                w = next(reversed(e.workspaces.values()))
                n = int(w["missing"].item())
                queue = w["ioids"][:n].cpu().tolist()
                assert queue == oracle(w, order, 16)
                assert len(set(queue)) == n
                assert w["ranks"][w["ioids"][:n].long()].cpu().tolist() == list(
                    range(n)
                )
                assert torch.isfinite(output).all()
                expected = selected_attention_oracle(
                    q.cpu(), backing, bb.cpu(), pp.cpu(), 16
                )
                # Independent mathematical check catches address/mask/multiplicity
                # bugs; the historical 0.03 is not a reorder performance gate.
                relative = (
                    (output.cpu().double() - expected).norm()
                    / expected.norm().clamp_min(1e-20)
                ).item()
                assert relative < 0.02
                stats = (
                    pool.read_bytes - before,
                    w["slots"].cpu().clone(),
                    w["all_ids"].cpu().clone(),
                    w["all_counts"].cpu().clone(),
                    w["valid"].cpu().clone(),
                )
                if order == "physical":
                    ref[repeat] = stats
                else:
                    assert stats[0] == ref[repeat][0]
                    for a, b in zip(stats[1:], ref[repeat][1:]):
                        assert torch.equal(a, b)
                for row in range(w["rows"]):
                    actual = [
                        int(v)
                        for st in range(w["stages"])
                        for v in w["ids"][st, row, : int(w["counts"][st, row])].cpu()
                    ]
                    assert Counter(actual) == Counter(
                        w["all_ids"][row, : int(w["all_counts"][row])].cpu().tolist()
                    )
                assert e.last_stats["read_order"] == order
            assert e.graphs.captures == 1
    finally:
        for e in executors:
            e.close()
        pool.close()


def test_read_order_rejects_unsupported_scope():
    e = GRSelectedTransport.__new__(GRSelectedTransport)
    e.read_order = "query_first_use"
    e.consumer_backend = "legacy"
    with pytest.raises(ValueError, match="silent fallback"):
        e.refusal(129, 8192, 2, 16, 16, 16)
