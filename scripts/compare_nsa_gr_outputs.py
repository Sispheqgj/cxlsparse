"""CPU-only comparison of benchmark_nsa_gr --save-outputs artifacts.

Accept benchmark roots (round-*/backend/variant/h-*) or individual h-* cases.
Cleanup checks inspect the recorded manifests; they never open a device or RPC.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_metrics(actual, reference, *, atol=0.03, rtol=0.03):
    result = dict(
        actual_shape=list(actual.shape),
        reference_shape=list(reference.shape),
        actual_dtype=str(actual.dtype),
        reference_dtype=str(reference.dtype),
        exact=False,
        allclose=False,
        elements=reference.numel(),
        outside_count=None,
        outside_fraction=None,
        max_abs=None,
        relative_l2=None,
    )
    if actual.shape != reference.shape:
        return result
    a, b = actual.double(), reference.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    error = (a - b).abs()
    outside = ~finite | (error > atol + rtol * b.abs())
    all_finite = bool(finite.all())
    result.update(
        exact=actual.dtype == reference.dtype and torch.equal(actual, reference),
        allclose=all_finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        actual_finite=bool(torch.isfinite(a).all()),
        reference_finite=bool(torch.isfinite(b).all()),
        different_count=int((a != b).sum()),
        outside_count=int(outside.sum()),
        outside_fraction=float(outside.double().mean()) if outside.numel() else 0.0,
        max_abs=float(error.max()) if all_finite and error.numel() else None,
        relative_l2=float((a - b).norm() / b.norm().clamp_min(1e-30))
        if all_finite
        else None,
    )
    return result


def recorded_cleanup(case, backend):
    name = "dax-backup.json" if backend == "cxl" else "pool-final-state.json"
    path = case / name
    if not path.exists():
        return dict(verified=False, path=str(path), reason="missing_manifest")
    data = json.loads(path.read_text())
    if backend == "cxl":
        regions = data.get("regions", []) if isinstance(data, dict) else []
        verified = bool(
            regions
            and data.get("restored") is True
            and all(region.get("restored") is True for region in regions)
        )
    elif backend == "rdma":
        verified = bool(
            isinstance(data, dict)
            and data.get("sessions") == 0
            and data.get("allocated_bytes") == 0
        )
    else:
        verified = False
    return dict(verified=verified, path=str(path), sha256=sha256(path), recorded=data)


def discover(root, variant):
    root = root.parent if root.name == "outputs" else root
    paths = (
        [root]
        if (root / "inputs.json").exists()
        else sorted(root.glob(f"round-*/*/{variant}/h-*"))
    )
    cases = {}
    for case in paths:
        worker = json.loads((case / "config.json").read_text())["worker"]
        round_name = next(
            (
                parent.name
                for parent in case.parents
                if parent.name.startswith("round-")
            ),
            "round-unspecified",
        )
        key = f"{round_name}/{worker['backend']}/h-{worker['max_history']}"
        if key in cases:
            raise ValueError(f"Duplicate comparison coordinate: {key}")
        cases[key] = case
    return cases


def compare_cases(actual_case, reference_case, *, atol=0.03, rtol=0.03):
    cases = dict(actual=actual_case, reference=reference_case)
    inputs, configs, metadata = {}, {}, {}
    errors = []
    for name, case in cases.items():
        input_path, config_path = case / "inputs.json", case / "config.json"
        inputs[name] = json.loads(input_path.read_text())
        configs[name] = json.loads(config_path.read_text())["worker"]
        metadata[name] = dict(
            path=str(case),
            inputs_sha256=sha256(input_path),
            canonical_inputs_sha256=canonical_sha256(inputs[name]),
            config_sha256=sha256(config_path),
            selected_backend=configs[name].get("nsa_selected_backend"),
            cleanup=recorded_cleanup(case, configs[name]["backend"]),
        )
        state_path = case / "run-state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            metadata[name]["run_state"] = dict(
                path=str(state_path), sha256=sha256(state_path), recorded=state
            )
            if not isinstance(state, dict) or state.get("completed") is not True:
                errors.append(f"{name}: run-state.completed is not true")
        else:
            metadata[name]["run_state"] = dict(
                path=str(state_path), present=False, policy="legacy_cleanup_only"
            )
        failed_path = case / "failed-run-cleanup.json"
        if failed_path.exists():
            failed = json.loads(failed_path.read_text())
            metadata[name]["failed_run_cleanup"] = dict(
                path=str(failed_path), sha256=sha256(failed_path), recorded=failed
            )
            if (
                failed.get("normal_cleanup_passed") is not True
                or failed.get("excluded_from_performance_acceptance") is True
            ):
                errors.append(f"{name}: failed-run cleanup excludes this case")
        if not metadata[name]["cleanup"]["verified"]:
            errors.append(f"{name}: cleanup not verified by recorded manifest")
    same_inputs = inputs["actual"] == inputs["reference"]
    if not same_inputs:
        errors.append("inputs.json content differs")
    for key in ("model", "model_path", "backend", "max_history"):
        if configs["actual"].get(key) != configs["reference"].get(key):
            errors.append(f"worker {key} differs")
    if configs["actual"].get("model") != "nsa_1b":
        errors.append("expected the real nsa_1b model")
    rows = []
    for q in sorted(set(inputs["actual"]) | set(inputs["reference"]), key=int):
        row = dict(extend=int(q), input_sha256={}, sources={}, errors=[])
        tensors = {}
        for name, case in cases.items():
            if q in inputs[name]:
                row["input_sha256"][name] = canonical_sha256(inputs[name][q])
            else:
                row["errors"].append(f"{name}: missing request")
            path = case / "outputs" / f"q-{q}.pt"
            if not path.exists():
                row["errors"].append(f"{name}: missing output")
                continue
            row["sources"][name] = dict(path=str(path), sha256=sha256(path))
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                row["errors"].append(f"{name}: output must be a 2D hidden tensor")
            elif len(tensor) != int(q):
                row["errors"].append(f"{name}: output row count differs from extend")
            else:
                tensors[name] = tensor
        row["same_input"] = (
            len(row["input_sha256"]) == 2
            and row["input_sha256"]["actual"] == row["input_sha256"]["reference"]
        )
        if len(tensors) == 2:
            row.update(tensor_metrics(**tensors, atol=atol, rtol=rtol))
        row["passed"] = bool(not errors and not row["errors"] and row.get("allclose"))
        rows.append(row)
    return dict(
        sources=metadata,
        same_inputs=same_inputs,
        identical_input_file_sha256=metadata["actual"]["inputs_sha256"]
        == metadata["reference"]["inputs_sha256"],
        errors=errors,
        outputs=rows,
        passed=bool(rows) and all(row["passed"] for row in rows),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--actual-variant", default="compact")
    parser.add_argument("--reference-variant", default="gpu_baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    roots = dict(
        actual=args.actual_root.resolve(),
        reference=(args.reference_root or args.actual_root).resolve(),
    )
    actual = discover(roots["actual"], args.actual_variant)
    reference = discover(roots["reference"], args.reference_variant)
    reports, errors = {}, []
    for key in sorted(set(actual) | set(reference)):
        if key not in actual or key not in reference:
            errors.append(f"Missing paired case: {key}")
            continue
        reports[key] = compare_cases(actual[key], reference[key])
    if not reports:
        errors.append("No paired cases found")
    provenance = {}
    for name, root in roots.items():
        path = root / "parameters.json"
        provenance[name] = dict(root=str(root))
        if path.exists():
            provenance[name].update(
                parameters=json.loads(path.read_text()), parameters_sha256=sha256(path)
            )
    passed = not errors and all(case["passed"] for case in reports.values())
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        actual_variant=args.actual_variant,
        reference_variant=args.reference_variant,
        tolerance=dict(atol=0.03, rtol=0.03),
        provenance=provenance,
        errors=errors,
        cases=reports,
        passed=passed,
        notes=[
            "CPU analysis only; cleanup is checked against recorded manifests, not live devices.",
            "Snapshots represent the first revisit at each length. Their new_length timings include recording overhead.",
            "Saved outputs do not establish correctness of every later steady/mixed request.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for key, case in reports.items():
        for row in case["outputs"]:
            print(
                key,
                f"Q={row['extend']}",
                f"exact={row.get('exact')}",
                f"allclose={row.get('allclose')}",
                f"outside={row.get('outside_count')}/{row.get('elements')}",
                f"max_abs={row.get('max_abs')}",
                f"relative_l2={row.get('relative_l2')}",
                f"passed={row['passed']}",
            )
    print(json.dumps(dict(passed=passed, errors=errors, report=str(args.output))))
    return 0 if passed or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
