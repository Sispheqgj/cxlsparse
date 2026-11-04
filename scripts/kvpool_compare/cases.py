"""Explicit treatments; ordinary controls own a separate KV target per query."""

CASES = {
    "rdma_batch_private": dict(
        backend="rdma",
        selected="triton_gr",
        shared=False,
        registration="per_call",
        label="RDMA 整批取回",
    ),
    "rdma_optimized_private": dict(
        backend="rdma",
        selected="triton_gr",
        shared=False,
        registration="persistent",
        label="RDMA 整批取回（注册复用）",
    ),
    "cxl_batch_private": dict(
        backend="cxl",
        selected="triton_gr",
        shared=False,
        registration="persistent",
        label="CXL 整批读取",
    ),
    "cxl_tma_private": dict(
        backend="cxl",
        selected="tilelang_tma",
        shared=False,
        registration="persistent",
        label="CXL 按块流水（不共享）",
    ),
    "cxl_tma_shared": dict(
        backend="cxl",
        selected="tilelang_tma",
        shared=True,
        registration="persistent",
        label="CXL 按块流水（共享，本方案）",
    ),
}


def folder(root, round_index, case, history):
    return (
        root / f"round-{round_index}" / CASES[case]["backend"] / case / f"h-{history}"
    )


def configure(bench, worker, case):
    spec = CASES[case]
    bench.VARIANTS[case] = (spec["selected"], 1)
    order = "consumption_depth" if spec["selected"] == "tilelang_tma" else "physical"
    worker.update(
        nsa_cross_query_dedup=spec["shared"],
        nsa_rdma_registration=spec["registration"],
        nsa_selected_read_order=order,
        nsa_selected_order="strict",
        nsa_selected_mode="overlap",
    )
    if spec["selected"] == "tilelang_tma":
        bench.TMA_READ_ORDERS[case] = order
        bench.READY_WINDOWS[case] = 1
