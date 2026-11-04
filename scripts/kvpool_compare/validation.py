"""CPU validation of replay, transfer accounting, and sharing treatments."""

import json
import math
from collections import Counter, defaultdict

from cases import CASES


def read(path):
    return json.loads(path.read_text())


def validate_case(path, parameters, source, manifest_digest):
    state, binding, config = (
        read(path / name)
        for name in ("run-state.json", "runtime-source.json", "config.json")
    )
    case = path.parent.name
    spec, worker = CASES[case], config["worker"]
    backend = spec["backend"]
    assert path.parents[1].name == backend
    assert (
        state["completed"]
        and not state["cleanup_errors"]
        and not state["forced_process_stops"]
    ), path
    assert binding == dict(
        runtime_snapshot=str(source),
        source_manifest_sha256=manifest_digest,
        backend=backend,
        variant=case,
    ), path
    assert (
        worker["backend"] == backend
        and worker["nsa_selected_backend"] == spec["selected"]
    ), path
    assert worker["nsa_cross_query_dedup"] is spec["shared"], path
    assert worker["nsa_rdma_registration"] == spec["registration"], path
    assert worker["nsa_selected_max_queries"] == 1024, path
    if backend == "cxl":
        backup = read(path / "dax-backup.json")
        assert (
            backup["restored"]
            and backup["regions"]
            and all(r["restored"] for r in backup["regions"])
        ), path
    else:
        pool = read(path / "pool-final-state.json")
        assert pool["sessions"] == pool["allocated_bytes"] == 0, path
    rows, samples, trace, previous, captures = (
        read(path / "results.json"),
        [],
        [],
        None,
        {},
    )
    counts = Counter()
    for position, row in enumerate(rows):
        result = row["response"]["result"]
        execution, pool, cache = (
            result["nsa_execution"],
            result["pool"],
            result["cache"],
        )
        assert row["backend"] == backend and row["variant"] == case, path
        assert row["history"] == worker["max_history"] and pool["backend"] == backend, (
            path
        )
        assert execution["selected_backend"] == spec["selected"], path
        delta = pool["read_bytes"] - (previous["pool"]["read_bytes"] if previous else 0)
        assert delta >= 0, path
        if backend == "rdma":
            assert (
                pool["transport"] == "gpudirect_dmabuf"
                and pool["cpu_staging_bytes"] == 0
            ), path
        if row["stage"] != "prefill":
            assert previous is not None, path
            assert result["prompt"]["input_tokens"] == row["extend"], path
            assert pool["write_bytes"] == row["history"] * 8192, path
            assert all(
                cache[name] >= 0
                for name in (
                    "hits",
                    "misses",
                    "old_references",
                    "deduplicated_references",
                )
            ), path
            assert delta == cache["misses"] * pool["token_bytes"], (
                path,
                delta,
                cache["misses"],
            )
            assert (
                cache["old_references"]
                == cache["hits"] + cache["misses"] + cache["deduplicated_references"]
            ), path
            # Private targets may still reuse repeated IDs within one query.
            # This counter therefore cannot alone identify cross-query sharing.
            old, transport = previous["nsa_execution"], execution["selected_transport"]
            assert (
                execution["selected_native_fallbacks"]
                == old["selected_native_fallbacks"]
            ), path
            assert execution["selected_gr_calls"] - old["selected_gr_calls"] == 16, path
            assert execution["selected_cross_query_dedup"] is spec["shared"], path
            assert (
                transport["cross_query_dedup"] is spec["shared"]
                and transport["effective_q"] == row["extend"]
            ), path
            assert transport["planner"] == (
                "active_gpu" if spec["shared"] else "query_private_gpu"
            ), path
            if backend == "rdma":
                assert transport["rdma_registration"] == spec["registration"], path
            if spec["selected"] == "tilelang_tma":
                assert (
                    execution["selected_tma_calls"] - old["selected_tma_calls"] == 16
                ), path
                assert (
                    transport["read_order"] == "consumption_depth"
                    and transport.get("ready_window", 1) == 1
                ), path
                assert transport["consumer"] == "tma_bounded_rows", path
            else:
                assert (
                    transport["consumer"] == "native_indexed_bounded"
                    and transport["io_policy"] == "batch_once"
                ), path
            assert all(
                math.isfinite(row[m]) and row[m] > 0 for m in ("worker_ms", "e2e_ms")
            ), path
            item = dict(
                row,
                read_bytes=delta,
                hits=cache["hits"],
                misses=cache["misses"],
                old_references=cache["old_references"],
                deduplicated_references=cache["deduplicated_references"],
                read_operations=pool["read_operations"]
                - previous["pool"]["read_operations"],
            )
            trace.append(item)
            counts[row["stage"], row["extend"]] += 1
            if row["stage"] in ("steady", "mixed"):
                samples.append(item)
        if row["stage"] == "warmup":
            captures[row["extend"]] = row["captures"]
        if row["stage"] == "steady":
            assert row["captures"] == captures[row["extend"]], path
        if row["stage"] == "mixed":
            assert row["captures"] == rows[position - 1]["captures"], path
        previous = result
    for length in parameters["lengths"]:
        expected = {
            "new_length": 1,
            "warmup": parameters["warmups"],
            "steady": parameters["repeats"],
            "mixed": 0 if parameters["skip_mixed"] else parameters["repeats"],
        }
        for stage, count in expected.items():
            assert counts[stage, length] == count, (
                path,
                stage,
                length,
                counts[stage, length],
                count,
            )
    return rows, samples, trace


def validate_traffic(traces):
    """Match every request, then require real additional reads for private KV."""
    coordinates = defaultdict(dict)
    for case, trace in traces:
        for position, row in enumerate(trace):
            key = row["round"], row["history"], position
            assert case not in coordinates[key]
            coordinates[key][case] = row
    fields = (
        "read_bytes",
        "hits",
        "misses",
        "old_references",
        "deduplicated_references",
    )
    checked, paired, points = 0, 0, defaultdict(list)
    for coordinate, cases in coordinates.items():
        signatures = {
            (r["stage"], r["extend"], r["response"]["result"]["prompt"]["input_tokens"])
            for r in cases.values()
        }
        assert len(signatures) == 1, (coordinate, "request sequences differ")
        private = [row for case, row in cases.items() if not CASES[case]["shared"]]
        if private:
            reference = tuple(private[0][f] for f in fields)
            for row in private[1:]:
                assert tuple(row[f] for f in fields) == reference, (
                    coordinate,
                    row["variant"],
                    "private traffic differs",
                )
                checked += 1
        shared = cases.get("cxl_tma_shared")
        if private and shared:
            p = cases.get("cxl_tma_private", private[0])
            assert p["old_references"] == shared["old_references"], (
                coordinate,
                "logical task changed",
            )
            assert (
                p["read_bytes"] >= shared["read_bytes"] and p["hits"] >= shared["hits"]
            ), coordinate
            paired += 1
            if p["stage"] in ("steady", "mixed"):
                points[p["stage"], p["history"], p["extend"]].append(
                    (p["read_bytes"], shared["read_bytes"])
                )
    reductions = []
    for (stage, history, extend), values in sorted(points.items()):
        private, shared = (sum(v[i] for v in values) for i in range(2))
        assert private > shared, (
            stage,
            history,
            extend,
            "no measured reduction in actual requested reads",
        )
        reductions.append(
            dict(
                stage=stage,
                history=history,
                extend=extend,
                requests=len(values),
                private_read_bytes=private,
                shared_read_bytes=shared,
                reduction_fraction=1 - shared / private,
                private_over_shared=private / shared if shared else None,
            )
        )
    return dict(
        passed=True,
        matched_private_comparisons=checked,
        shared_private_request_pairs=paired,
        reductions=reductions,
        scope="Software-accounted external payload reads; not physical link traffic. Runtime diagnostics check the private planner route; target non-aliasing is a separate planner correctness check.",
    )
