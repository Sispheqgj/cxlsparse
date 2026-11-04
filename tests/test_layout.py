import importlib.util
import subprocess
import sys
from pathlib import Path

from cxl_recsys.common.config import DEFAULT_CONFIG
from cxl_recsys.common.paths import PROJECT_ROOT
from cxl_recsys.mempool import rdma


def test_resource_paths_after_reorganization(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert DEFAULT_CONFIG == PROJECT_ROOT / "config.yml"
    assert DEFAULT_CONFIG.is_file()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert Path(rdma.__file__).with_name("native").joinpath("verbs.c").is_file()

    for module in (
        "pool",
        "rdma",
        "request",
        "rpc",
        "config",
        "workload",
        "engine",
        "hisparse",
    ):
        assert importlib.util.find_spec(f"cxl_recsys.{module}") is None


def test_scheduler_import_does_not_load_gpu_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cxl_recsys.scheduler; "
            "import cxl_recsys.scheduler.workload; "
            "assert 'torch' not in sys.modules; "
            "import cxl_recsys.models; "
            "assert 'torch' not in sys.modules; "
            "assert 'cxl_recsys.models.simulation.model' not in sys.modules; "
            "assert 'cxl_recsys.models.simulation.deepseek' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
