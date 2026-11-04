"""CPU-only validation and aggregation of the five matched system configurations."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from cases import folder
from evidence import check_snapshot, digest, write
from validation import validate_case, validate_traffic


def read(path):
    return json.loads(path.read_text())


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def analyze(root, candidate):
    parameters, complete = read(root / "parameters.json"), read(root / "complete.json")
    assert complete["passed"] is True
    external = read(root / "external-fingerprints-verified.json")
    assert external["passed"] and external["original_fingerprints_sha256"] == digest(
        root / "input-model-fingerprints.json"
    )
    assert not any(
        parameters.get(name)
        for name in ("order_probe", "window_probe", "prepare_probe")
    )
    source = Path(read(root / "source.json")["path"])
    manifest = check_snapshot(source)
    assert manifest == read(root / "source-sha256.json")
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    assert candidate in parameters["cases"]
    expected = {
        str(folder(root, r, case, h).relative_to(root))
        for r in range(parameters["rounds"])
        for case in parameters["cases"]
        for h in parameters["histories"]
    }
    found = {str(p.relative_to(root)) for p in root.glob("round-*/*/*/h-*")}
    assert found == expected, (found - expected, expected - found)
    assert len(found) == complete["cases"]
    inputs, records, samples, traces, settings = {}, {}, [], [], None
    for name in sorted(found):
        path = root / name
        config = read(path / "config.json")["worker"]
        common = {
            k: config[k]
            for k in (
                "model",
                "model_path",
                "hot",
                "page_size",
                "max_users",
                "max_batch_size",
                "nsa_cuda_graphs",
                "nsa_graph_cache_size",
                "nsa_kernel_backend",
            )
        }
        settings = settings or common
        assert settings == common, (name, "model/cache settings differ")
        coordinate = (path.parents[2].name, path.name)
        request = read(path / "inputs.json")
        if coordinate in inputs:
            assert inputs[coordinate] == request, (name, "inputs differ")
        inputs[coordinate] = request
        rows, measured, trace = validate_case(path, parameters, source, manifest_digest)
        traces.append((path.parent.name, trace))
        records[name] = rows
        samples.extend(measured)
    traffic = validate_traffic(traces)
    write(root / "traffic-validation.json", traffic)
    ordered = []
    for item in read(root / "run-order.json"):
        ordered.extend(
            records[
                f"round-{item['round']}/{item['backend']}/{item['variant']}/h-{item['history']}"
            ]
        )
    assert (
        ordered == read(root / "results.json") and len(ordered) == complete["samples"]
    )
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["stage"], row["history"], row["extend"], row["variant"]].append(row)
    numerical = (
        read(root / "correctness.json")
        if (root / "correctness.json").exists()
        else None
    )
    numerical_status = defaultdict(list)
    if numerical:
        assert numerical["parameters_sha256"] == digest(root / "parameters.json")
        for case in numerical["cases"]:
            for row in case["outputs"]:
                numerical_status[case["history"], row["extend"], case["actual"]].append(
                    row["passed"]
                )
                numerical_status[
                    case["history"], row["extend"], case["reference"]
                ].append(row.get("reference_finite") is True)
    summary = []
    for (stage, history, extend, case), rows in sorted(grouped.items()):
        assert len(rows) == parameters["rounds"] * parameters["repeats"]
        summary.append(
            dict(
                stage=stage,
                history=history,
                extend=extend,
                case=case,
                samples=len(rows),
                numerical_accepted=(
                    all(numerical_status[history, extend, case])
                    if numerical_status[history, extend, case]
                    else None
                ),
                **{
                    f"{metric}_p{percent}": quantile(
                        [r[metric] for r in rows], percent / 100
                    )
                    for metric in ("worker_ms", "e2e_ms")
                    for percent in (10, 50, 90)
                },
                **{
                    f"{metric}_median": statistics.median(r[metric] for r in rows)
                    for metric in (
                        "read_bytes",
                        "read_operations",
                        "hits",
                        "misses",
                        "old_references",
                        "deduplicated_references",
                    )
                },
                per_round={
                    str(i): {
                        metric: statistics.median(
                            r[metric] for r in rows if r["round"] == i
                        )
                        for metric in ("worker_ms", "e2e_ms")
                    }
                    for i in range(parameters["rounds"])
                },
            )
        )
    index = {(r["stage"], r["history"], r["extend"], r["case"]): r for r in summary}
    comparisons = []
    for row in summary:
        if row["case"] == candidate:
            continue
        target = index[row["stage"], row["history"], row["extend"], candidate]
        comparisons.append(
            dict(
                stage=row["stage"],
                history=row["history"],
                extend=row["extend"],
                baseline=row["case"],
                accepted=row["numerical_accepted"] is True
                and target["numerical_accepted"] is True,
                worker_speedup=row["worker_ms_p50"] / target["worker_ms_p50"],
                e2e_speedup=row["e2e_ms_p50"] / target["e2e_ms_p50"],
                read_bytes_ratio=(
                    row["read_bytes_median"] / target["read_bytes_median"]
                    if target["read_bytes_median"]
                    else None
                ),
                per_round_e2e_speedup={
                    i: row["per_round"][i]["e2e_ms"] / target["per_round"][i]["e2e_ms"]
                    for i in row["per_round"]
                },
            )
        )
    means = []
    for stage, baseline in sorted({(r["stage"], r["baseline"]) for r in comparisons}):
        values = [
            r for r in comparisons if (r["stage"], r["baseline"]) == (stage, baseline)
        ]
        accepted = all(r["accepted"] for r in values)
        means.append(
            dict(
                stage=stage,
                baseline=baseline,
                points=len(values),
                accepted=accepted,
                **{
                    metric: math.exp(
                        statistics.mean(math.log(r[metric]) for r in values)
                    )
                    if accepted
                    else None
                    for metric in ("worker_speedup", "e2e_speedup")
                },
            )
        )
    report = dict(
        passed=bool(numerical and numerical["passed"]),
        performance_valid=True,
        numerical_valid=numerical["passed"] if numerical else None,
        candidate=candidate,
        cases=sorted(found),
        traffic=traffic,
        summary=summary,
        comparisons=comparisons,
        geometric_means=means,
        notes=[
            "E2E includes Scheduler-to-Worker RPC and full forward, excluding model load and history construction.",
            "Only steady/mixed requests are aggregated; saved first-revisit output timing is excluded.",
            "RDMA is a same-host GPUDirect loopback Pool; it is not a two-host network experiment.",
            "Read bytes are software-requested payload, not measured CXL/PCIe/RDMA link traffic.",
            "All private cases keep per-query KV targets; the two RDMA cases differ only in per-call versus persistent HBM registration. All cases retain hot replicas and missing-only fetch.",
            "Performance validation is separate from full-hidden numerical comparison in correctness.json.",
        ],
    )
    write(root / "summary.json", report)
    for name, rows in (("summary", summary), ("comparisons", comparisons)):
        if rows:
            with (root / f"{name}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = [
        "| Stage | H | E | Configuration | Numerical accepted | Worker p50 ms | E2E p50 ms | Read MiB/request |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['stage']} | {row['history']} | {row['extend']} | {row['case']} | "
            f"{row['numerical_accepted']} | {row['worker_ms_p50']:.3f} | {row['e2e_ms_p50']:.3f} | {row['read_bytes_median'] / 2**20:.3f} |"
        )
    (root / "results-table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(means, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate", default="cxl_tma_shared")
    args = parser.parse_args()
    raise SystemExit(0 if analyze(args.root.resolve(), args.candidate)["passed"] else 1)
