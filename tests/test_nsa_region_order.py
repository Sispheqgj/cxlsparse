"""Independent queue oracle: fresh residency/selection/device priority on graph replay."""

from collections import Counter

import pytest
import torch

from cxl_recsys.models.nsa.planning.active import _ranks
from cxl_recsys.models.nsa.planning.read_order import (
    allocate_read_order,
    allocate_region_order,
    reorder_reads,
)


@pytest.mark.gpu
@pytest.mark.parametrize("regions", [1, 2, 3])
def test_region_first_use_graph(regions):
    p, bs, rows, width = 19, 16, 70, 6
    w = dict(
        physical=p, bs=bs, rows=rows, heads=2, per_head=10, width=width, scratch_bytes=0
    )
    for k, shape, dtype in [
        ("positions", (rows,), torch.int32),
        ("all_ids", (rows, width), torch.int32),
        ("all_counts", (rows,), torch.int32),
        ("flags", (p,), torch.int32),
        ("slots", (p * bs,), torch.int64),
        ("valid", (p,), torch.int32),
        ("ioids", (p,), torch.int32),
        ("ranks", (p,), torch.int32),
        ("missing", (1,), torch.int32),
    ]:
        w[k] = torch.empty(shape, dtype=dtype, device="cuda")
    allocate_read_order(w, "cuda")
    allocate_region_order(w, "cuda", regions, 16)

    def plan():
        _ranks[(1,)](
            w["flags"], w["ioids"], w["ranks"], w["missing"], p, 32, num_warps=4
        )
        reorder_reads(w, "query_first_use_region", 16)

    def populate(seed):
        g = torch.Generator().manual_seed(seed)
        pos = torch.arange(35).repeat_interleave(2).int()
        pos[14:16] = -1
        ids = torch.randint(0, p, (rows, width), generator=g, dtype=torch.int32)
        counts = torch.full((rows,), width, dtype=torch.int32)
        counts[14:16] = 0
        # Valid duplicates, no-new-block middle group, partial last group.
        ids[:, 1] = ids[:, 0]
        ids[32:64] = ids[:32]
        valid = torch.randint(1, 17, (p,), generator=g, dtype=torch.int32)
        slots = torch.arange(p * bs).reshape(p, bs).long()
        slots[torch.arange(bs)[None, :] >= valid[:, None]] = -1
        if seed == 1:
            slots[slots >= 0] = (slots[slots >= 0] // 16 * regions) * 16 + slots[
                slots >= 0
            ] % 16  # only region zero
        if seed == 2:
            slots[::2] = -1
        if seed == 3:
            slots[:] = -1  # no external reads
        # Mixed-source partial block, including unequal effective device bytes.
        if seed == 4 and regions > 1:
            slots[0, :] = torch.tensor([0] * 5 + [16] * 11)
            valid[0] = 13
        flags = (slots >= 0).any(1).int()
        for k, v in dict(
            positions=pos,
            all_ids=ids,
            all_counts=counts,
            slots=slots.flatten(),
            valid=valid,
            flags=flags,
        ).items():
            w[k].copy_(v)

    populate(0)
    for _ in range(3):
        plan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    for seed in [0, 1, 2, 3, 4, 0]:
        populate(seed)
        graph.replay()
        torch.cuda.synchronize()
        ids, counts, pos, flags, slots, valid = [
            w[k].cpu()
            for k in ["all_ids", "all_counts", "positions", "flags", "slots", "valid"]
        ]
        slots = slots.reshape(p, bs)
        first = {}
        token = -1
        for row in range(rows):
            if row % 2 == 0 and pos[row] >= 0:
                token += 1
            if pos[row] < 0:
                continue
            for b in ids[row, : counts[row]].tolist():
                if flags[b]:
                    first[b] = min(first.get(b, 99), token // 16)
        regions_of = {}
        for b in range(p):
            c = Counter(
                (int(s) // 16) % regions for s in slots[b, : valid[b]] if s >= 0
            )
            regions_of[b] = max(range(regions), key=lambda r: (c[r], -r))
        expected = []
        for group in range(4):
            sub = [
                [
                    b
                    for b in range(p)
                    if flags[b]
                    and min(first.get(b, 3), 3) == group
                    and regions_of[b] == r
                ]
                for r in range(regions)
            ]
            for k in range(max(map(len, sub), default=0)):
                expected.extend(s[k] for s in sub if k < len(s))
        n = int(w["missing"])
        queue = w["ioids"][:n].cpu().tolist()
        assert queue == expected
        assert len(set(queue)) == int(flags.sum()) == n
        assert w["ranks"][w["ioids"][:n].long()].cpu().tolist() == list(range(n))
        assert (w["ioids"][n:] == -1).all() and (
            w["ranks"][~flags.bool().cuda()] == -1
        ).all()


@pytest.mark.gpu
@pytest.mark.parametrize("residency", ["miss", "partial", "hit", "empty"])
def test_region_mode_production_attention(residency):
    from test_nsa_read_order import oracle
    from test_nsa_transport import setup_case

    from cxl_recsys.models.nsa.backends.io_plan import selected_attention_oracle
    from cxl_recsys.models.nsa.transport import GRSelectedTransport

    pool, cache, backing, q, pos, blocks, fetches = setup_case(residency)
    executor = GRSelectedTransport(
        cache,
        consume_order="block_ready",
        read_order="query_first_use_region",
        read_group_size=16,
        chunk_blocks=2,
        io_ctas=4,
        cuda_graphs=True,
    )
    initial = {k: getattr(cache, k).clone() for k in ("tags", "locs", "data")}
    try:
        for repeat in range(3):
            for k, v in initial.items():
                getattr(cache, k).copy_(v)
            bb = blocks.roll(repeat, 0).contiguous()
            out, _ = executor.run(q, bb, pos, fetches, 16)
            executor.last_done.synchronize()
            w = next(reversed(executor.workspaces.values()))
            n = int(w["missing"])
            assert w["ioids"][:n].cpu().tolist() == oracle(w, "query_first_use", 16)
            assert torch.isfinite(out).all()
            expected = selected_attention_oracle(
                q.cpu(), backing, bb.cpu(), pos.cpu(), 16
            )
            assert (out.cpu().double() - expected).norm() / expected.norm().clamp_min(
                1e-20
            ) < 0.02
            for row in range(w["rows"]):
                actual = [
                    int(v)
                    for st in range(w["stages"])
                    for v in w["ids"][st, row, : int(w["counts"][st, row])].cpu()
                ]
                assert Counter(actual) == Counter(
                    w["all_ids"][row, : int(w["all_counts"][row])].cpu().tolist()
                )
            assert executor.last_stats["read_order"] == "query_first_use_region"
        assert executor.graphs.captures == 1
    finally:
        executor.close()
        pool.close()
