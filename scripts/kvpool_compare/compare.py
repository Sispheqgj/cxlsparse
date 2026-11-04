"""Compare saved full hidden tensors, explicitly allowing CXL/RDMA backend changes."""

import argparse
import json
import sys
from pathlib import Path

from cases import folder as case_folder
from evidence import check_snapshot, digest, write

# These are the intended treatment and per-process routing fields. Every other
# Worker configuration field, input token and shared source remains matched.
TREATMENT_FIELDS = {
    "backend",
    "listen",
    "log_dir",
    "pool_target",
    "nsa_selected_backend",
    "nsa_selected_max_queries",
    "nsa_selected_query_group",
    "nsa_selected_workspace_mib",
    "nsa_selected_read_order",
    "nsa_selected_ready_window",
    "nsa_cross_query_dedup",
    "nsa_rdma_registration",
}


def compare(root, reference):
    import torch

    torch.set_num_threads(4)
    parameters = json.loads((root / "parameters.json").read_text())
    source = Path(json.loads((root / "source.json").read_text())["path"])
    check_snapshot(source)
    sys.path.insert(0, str(source))
    from scripts.compare_nsa_gr_outputs import compare_cases

    assert reference in parameters["cases"]
    reports = []
    for round_index in range(parameters["rounds"]):
        for history in parameters["histories"]:
            for case in parameters["cases"]:
                if case == reference:
                    continue
                actual = case_folder(root, round_index, case, history)
                expected = case_folder(root, round_index, reference, history)
                configs = [
                    json.loads((folder / "config.json").read_text())["worker"]
                    for folder in (actual, expected)
                ]
                unexpected = {
                    key: [config.get(key) for config in configs]
                    for key in configs[0].keys() | configs[1].keys()
                    if key not in TREATMENT_FIELDS
                    and configs[0].get(key) != configs[1].get(key)
                }
                bindings = [
                    json.loads((folder / "runtime-source.json").read_text())
                    for folder in (actual, expected)
                ]
                same_source = all(
                    bindings[0][key] == bindings[1][key]
                    for key in ("runtime_snapshot", "source_manifest_sha256")
                )
                report = compare_cases(actual, expected)
                backend_change = configs[0]["backend"] != configs[1]["backend"]
                if backend_change:
                    assert {configs[0]["backend"], configs[1]["backend"]} == {
                        "cxl",
                        "rdma",
                    }
                    # The reused validator checks backend equality by default.
                    # Remove only this explicit, intended difference; its input,
                    # model, cleanup and tensor checks remain unchanged.
                    report["errors"] = [
                        e for e in report["errors"] if e != "worker backend differs"
                    ]
                if unexpected:
                    report["errors"].append(
                        f"Unexpected configuration differences: {unexpected}"
                    )
                if not same_source:
                    report["errors"].append("Runtime source differs")
                for folder in (actual, expected):
                    if (
                        json.loads((folder / "run-state.json").read_text()).get(
                            "completed"
                        )
                        is not True
                    ):
                        report["errors"].append(f"Incomplete run: {folder}")
                for row in report["outputs"]:
                    row["passed"] = bool(
                        not report["errors"]
                        and not row["errors"]
                        and row.get("allclose")
                    )
                report.update(
                    passed=bool(report["outputs"])
                    and all(r["passed"] for r in report["outputs"]),
                    round=round_index,
                    history=history,
                    actual=case,
                    reference=reference,
                    explicitly_allowed_backend_change=backend_change,
                    same_runtime_source=same_source,
                )
                reports.append(report)
    result = dict(
        passed=bool(reports) and all(r["passed"] for r in reports),
        reference=reference,
        tolerance=dict(atol=0.03, rtol=0.03),
        cases=reports,
        parameters_sha256=digest(root / "parameters.json"),
        scope="Full hidden outputs from first revisit at each Extend; no inference about every later request.",
    )
    write(root / "correctness.json", result)
    for report in reports:
        for row in report["outputs"]:
            print(
                report["actual"],
                f"H={report['history']} E={row['extend']}",
                f"exact={row.get('exact')} allclose={row.get('allclose')} max_abs={row.get('max_abs')}",
            )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", default="cxl_batch_private")
    args = parser.parse_args()
    raise SystemExit(compare(args.root.resolve(), args.reference))
