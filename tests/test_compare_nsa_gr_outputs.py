"""Saved-output checks must reject mismatched inputs and incomplete cleanup."""

import json

import pytest
import torch

from scripts.compare_nsa_gr_outputs import compare_cases, main, tensor_metrics


def case(root, variant, backend="rdma"):
    path = root / "round-0" / backend / variant / "h-80"
    (path / "outputs").mkdir(parents=True)
    (path / "inputs.json").write_text(
        json.dumps({"2": {"history_token_ids": [1, 2], "candidate_token_ids": [3, 4]}})
    )
    (path / "config.json").write_text(
        json.dumps(
            dict(
                worker=dict(
                    model="nsa_1b",
                    model_path="checkpoint",
                    backend=backend,
                    max_history=80,
                    nsa_selected_backend="tilelang_gr"
                    if variant == "compact"
                    else "tilelang_gpu",
                )
            )
        )
    )
    if backend == "rdma":
        (path / "pool-final-state.json").write_text(
            json.dumps(dict(sessions=0, allocated_bytes=0))
        )
    else:
        (path / "dax-backup.json").write_text(
            json.dumps(dict(restored=True, regions=[dict(restored=True)]))
        )
    torch.save(torch.ones((2, 3), dtype=torch.bfloat16), path / "outputs/q-2.pt")
    return path


def test_metrics_use_reference_scaled_tolerance_and_reject_nonfinite():
    reference = torch.tensor([1.0, 100.0, 0.0])
    actual = torch.tensor([1.07, 102.0, 0.031])
    result = tensor_metrics(actual, reference)
    assert not result["exact"] and not result["allclose"]
    assert result["outside_count"] == 2
    assert result["max_abs"] == 2
    assert result["relative_l2"] > 0
    result = tensor_metrics(torch.tensor([float("nan")]), torch.tensor([0.0]))
    assert result["outside_count"] == 1
    assert result["max_abs"] is None and not result["allclose"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("backend", ["cxl", "rdma"])
def test_matching_inputs_outputs_and_recorded_cleanup(tmp_path, backend):
    actual, reference = (
        case(tmp_path, variant, backend) for variant in ("compact", "gpu_baseline")
    )
    result = compare_cases(actual, reference)
    assert result["passed"] and result["same_inputs"]
    assert result["identical_input_file_sha256"]
    assert result["outputs"][0]["exact"]
    # Different formatting remains visible in raw hashes, but inputs are equal.
    data = json.loads((reference / "inputs.json").read_text())
    (reference / "inputs.json").write_text(json.dumps(data, indent=2))
    result = compare_cases(actual, reference)
    assert result["passed"] and not result["identical_input_file_sha256"]


def test_equal_output_is_not_evidence_for_different_requests(tmp_path):
    actual, reference = (
        case(tmp_path, variant) for variant in ("compact", "gpu_baseline")
    )
    data = json.loads((actual / "inputs.json").read_text())
    data["2"]["candidate_token_ids"][0] = 99
    (actual / "inputs.json").write_text(json.dumps(data))
    result = compare_cases(actual, reference)
    assert result["outputs"][0]["exact"]
    assert not result["outputs"][0]["same_input"]
    assert not result["passed"]


def test_missing_output_and_unreleased_pool_fail_cli(tmp_path):
    actual = case(tmp_path, "compact")
    case(tmp_path, "gpu_baseline")
    (actual / "outputs/q-2.pt").unlink()
    (actual / "pool-final-state.json").write_text(
        json.dumps(dict(sessions=1, allocated_bytes=1))
    )
    output = tmp_path / "report.json"
    assert main(["--actual-root", str(tmp_path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["passed"]
    result = report["cases"]["round-0/rdma/h-80"]
    assert not result["sources"]["actual"]["cleanup"]["verified"]
    assert result["outputs"][0]["errors"] == ["actual: missing output"]


@pytest.mark.parametrize("marker", ["run-state.json", "failed-run-cleanup.json"])
def test_failed_run_is_excluded_even_if_outputs_and_pool_match(tmp_path, marker):
    actual, reference = (
        case(tmp_path, variant) for variant in ("compact", "gpu_baseline")
    )
    (actual / marker).write_text(
        json.dumps(dict(completed=False, normal_cleanup_passed=False))
    )
    result = compare_cases(actual, reference)
    assert result["outputs"][0]["exact"]
    assert result["sources"]["actual"]["cleanup"]["verified"]
    assert not result["passed"]
