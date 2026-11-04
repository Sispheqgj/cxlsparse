import pytest
import yaml
from batch_helpers import batch_methods

from cxl_recsys.cli import parse_args
from cxl_recsys.common.config import BUILD_WORKER_DEFAULTS, DEFAULT_CONFIG
from cxl_recsys.common.sizes import dax_values


def command_defaults(command):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    defaults = config[command]
    if command == "worker":
        for key in ("dax_device", "dax_offset", "dax_bytes"):
            defaults[key] = dax_values(key, defaults[key])
    if command == "build-kernels":
        for key in BUILD_WORKER_DEFAULTS:
            defaults.setdefault(key, config["worker"][key])
    return defaults


@pytest.mark.parametrize("command", ["scheduler", "worker", "pool", "build-kernels"])
def test_every_option_uses_yaml(command):
    defaults = command_defaults(command)
    args = parse_args([command])
    assert {k: v for k, v in vars(args).items() if k != "command"} == defaults


def test_default_cxl_regions_use_only_dax0():
    args = parse_args(["worker", "--backend", "cxl"])
    assert args.dax_device == ["/dev/dax0.0", "/dev/dax0.0"]
    assert args.dax_offset == [0, 128 * 1024**3]
    assert args.dax_bytes == [128 * 1024**3, 128 * 1024**3]


@pytest.mark.parametrize("command", ["scheduler", "worker", "pool", "build-kernels"])
def test_all_options_can_be_overridden_in_yaml(tmp_path, command):
    original = command_defaults(command)
    changed = {}
    for key, value in original.items():
        if key == "dax_device":
            changed[key] = ["/dev/dax1.0"]
        elif key in ("dax_offset", "dax_bytes"):
            changed[key] = [2097152]
        elif key == "model":
            changed[key] = "nsa_1b"
        elif key == "nsa_kernel_backend":
            changed[key] = "torch"
        elif key == "nsa_selected_backend":
            changed[key] = "tilelang"
        elif key == "nsa_rdma_registration":
            changed[key] = "per_call"
        elif key == "nsa_selected_read_order":
            changed[key] = "query_first_use"
        elif key == "nsa_selected_order":
            changed[key] = "resident_first"
        elif key == "nsa_selected_mode":
            changed[key] = "serial"
        elif key == "nsa_cache_backend":
            changed[key] = "reference"
        elif key == "dsa_linear_backend":
            changed[key] = "bf16"
        elif key == "dsa_cache_backend":
            changed[key] = "reference"
        elif key in ("nosa_attention_backend", "nosa_indexer_backend"):
            changed[key] = "sm90"
        elif key == "nosa_cache_backend":
            changed[key] = "device"
        elif key == "nosa_selection_layout":
            changed[key] = "block"
        elif key == "nosa_compute_backend":
            changed[key] = "flashinfer"
        elif key == "log_level":
            changed[key] = "INFO"
        elif key in ("backend", "source", "sampling"):
            changed[key] = {"backend": "cxl", "source": "requests", "sampling": "zipf"}[
                key
            ]
        elif key in (
            "gr_prompt_format",
            "gr_heat_source",
            "gr_industrial_heat_field",
            "gr_text_material",
            "gr_text_dataset",
            "gr_arrival",
            "gr_sampling",
        ):
            changed[key] = {
                "gr_prompt_format": "nosa",
                "gr_heat_source": "industrial",
                "gr_industrial_heat_field": "pv_int",
                "gr_text_material": "synthetic",
                "gr_text_dataset": "games",
                "gr_arrival": "constant",
                "gr_sampling": "sequential",
            }[key]
        elif key in ("gr_user_probabilities", "gr_item_probabilities"):
            changed[key] = [1.0]
        elif type(value) is bool:
            changed[key] = not value
        elif type(value) in (int, float):
            changed[key] = value + 1
        elif isinstance(value, list):
            changed[key] = (
                [value[0] + 1] if type(value[0]) is int else ["remote-host:5051"]
            )
        elif key in ("dax_offset", "dax_bytes"):
            changed[key] = 2097152
        else:
            changed[key] = "changed-value"
    path = tmp_path / "all-options.yml"
    path.write_text(yaml.safe_dump({command: changed}))
    args = parse_args([command, "--config", str(path)])
    assert {
        k: v for k, v in vars(args).items() if k not in ("command", "config")
    } == changed


def test_override_and_explicit_cli(tmp_path):
    path = tmp_path / "custom.yml"
    path.write_text(
        'scheduler:\n  users: 7\n  seed: 99\n  workers: ["host-a:1", "host-b:2"]\nworker:\n  layers: 3\n  dax_offset: 2097152\n'
    )
    for argv in (
        ["--config", str(path), "scheduler"],
        ["scheduler", "--config", str(path)],
    ):
        args = parse_args([*argv, "--users", "3", "--seed", "0"])
        assert args.users == 3
        assert args.seed == 0
        assert args.workers == ["host-a:1", "host-b:2"]
        assert args.num_requests == 8
    assert parse_args(["worker", "--config", str(path)]).layers == 3
    assert parse_args(["worker", "--config", str(path), "--layers", "2"]).layers == 2
    assert parse_args(["worker", "--config", str(path)]).dax_offset == [2097152]


@pytest.mark.parametrize(
    "content",
    [
        "worker:\n  backend: invalid\n",
        "worker:\n  nsa_selected_ready_window: 3\n",
        "worker:\n  no_verify: 'false'\n",
        "worker:\n  timing_breakdown: 'false'\n",
        "worker:\n  timing_breakdown: 1\n",
        "worker:\n  timing_breakdown: null\n",
        "scheduler:\n  users: true\n",
        "scheduler:\n  workers: host-a:1\n",
        "scheduler:\n  users: null\n",
        "worker:\n  typo: 1\n",
        "scheduler:\n  seed: 1\n  seed: 2\n",
        "unknown: {}\n",
        "scheduler:\n  run_timeout: .nan\n",
        "worker: []\n",
    ],
)
def test_invalid_config_rejected(tmp_path, content):
    path = tmp_path / "bad.yml"
    path.write_text(content)
    with pytest.raises(SystemExit) as exc:
        parse_args(["worker", "--config", str(path)])
    assert exc.value.code == 2


def test_missing_file_and_removed_verification_switches(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["scheduler", "--config", str(tmp_path / "missing.yml")])
    for removed in ("--verify", "--no-verify"):
        with pytest.raises(SystemExit):
            parse_args(["worker", removed])


def test_tma_ready_window_cli_overrides_yaml(tmp_path):
    path = tmp_path / "ready-window.yml"
    path.write_text("worker:\n  nsa_selected_ready_window: 2\n")
    args = ["worker", "--config", str(path)]
    assert parse_args(["worker"]).nsa_selected_ready_window == 1
    assert parse_args(args).nsa_selected_ready_window == 2
    assert parse_args([*args, "--nsa-selected-ready-window", "4"]).nsa_selected_ready_window == 4
    with pytest.raises(SystemExit):
        parse_args([*args, "--nsa-selected-ready-window", "3"])


def test_timing_breakdown_boolean_cli_overrides_yaml(tmp_path):
    assert parse_args(["worker"]).timing_breakdown is False
    assert parse_args(["worker", "--timing-breakdown"]).timing_breakdown is True
    path = tmp_path / "timing.yml"
    path.write_text("worker:\n  timing_breakdown: true\n")
    args = ["worker", "--config", str(path)]
    assert parse_args(args).timing_breakdown is True
    assert parse_args([*args, "--no-timing-breakdown"]).timing_breakdown is False


def test_scheduler_runs_from_config_only(tmp_path):
    import json
    import socket
    import subprocess
    import sys

    from cxl_recsys.common.rpc import serve

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        target = f"127.0.0.1:{sock.getsockname()[1]}"
    received = {}

    def submit(request):
        received[request["request_id"]] = request
        return {"accepted": True, "epoch": "config-test"}

    def get(request):
        record = received[request["request_id"]]
        return {
            "status": "SUCCEEDED",
            "epoch": "config-test",
            "request_id": record["request_id"],
            "user_state": {
                "kv_length": len(record["history_token_ids"]),
                "backend": "dram",
            },
        }

    server = serve(
        target,
        batch_methods(submit, get, lambda _: {"epoch": "config-test"}),
    )
    config, output = tmp_path / "run.yml", tmp_path / "result.json"
    config.write_text(
        yaml.safe_dump(
            {
                "scheduler": {
                    "workers": [target],
                    "output": str(output),
                    "source": "generated",
                    "num_requests": 2,
                    "users": 1,
                    "history_length": 3,
                    "max_pending": 1,
                    "run_timeout": 10,
                    "log_dir": str(tmp_path / "logs"),
                }
            }
        )
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cxl_recsys.cli",
                "scheduler",
                "--config",
                str(config),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["state"] == "COMPLETED"
        log_files = list((tmp_path / "logs").glob("scheduler-*.log"))
        assert len(log_files) == 1
        log_text = log_files[0].read_text()
        for event in ("workload_loaded", "request_dispatched", "sequence_finished"):
            assert event in log_text
            assert event in result.stderr
        assert json.loads(output.read_text())["succeeded"] == 2
        assert len(received) == 2
        assert all(len(r["history_token_ids"]) == 3 for r in received.values())
    finally:
        server.stop(0).wait()


def test_nsa_query_private_controls(tmp_path):
    path = tmp_path / "query-private.yml"
    path.write_text(
        "worker:\n  nsa_selected_backend: triton_gr\n"
        "  nsa_cross_query_dedup: false\n  nsa_rdma_registration: per_call\n"
    )
    args = parse_args(["worker", "--config", str(path)])
    assert args.nsa_selected_backend == "triton_gr"
    assert not args.nsa_cross_query_dedup
    assert args.nsa_rdma_registration == "per_call"
    assert parse_args(["worker"]).nsa_cross_query_dedup
    assert not parse_args(["worker", "--no-nsa-cross-query-dedup"]).nsa_cross_query_dedup
