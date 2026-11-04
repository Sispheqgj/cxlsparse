import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from cxl_recsys.common.logging import (
    close_logging,
    configure_logging,
    prune_logs,
    request_context,
)


@pytest.mark.parametrize("role", ["scheduler", "worker", "mempool"])
def test_console_file_context_and_reconfiguration(tmp_path, capsys, role):
    logger = logging.getLogger(f"cxl_recsys.{role}.test")
    try:
        first = configure_logging(role, tmp_path, "DEBUG", 4096, 2)
        with request_context("r1", "u1"):
            logger.debug("pages_ready count=2")
        logger.debug("outside_request")
        content = first.read_text()
        assert "DEBUG" in content
        for output in (content, capsys.readouterr().err):
            assert "request_id='r1' user_id='u1' pages_ready" in output
            for message in ("logging_started", "outside_request"):
                line = next(line for line in output.splitlines() if message in line)
                assert "request_id=" not in line
                assert "user_id=" not in line
        second = configure_logging(role, tmp_path, "DEBUG", 4096, 2)
        logger.debug("unique_message")
        assert second != first
        assert second.read_text().count("unique_message") == 1
        assert "unique_message" not in first.read_text()
        assert len(logging.getLogger("cxl_recsys").handlers) == 2
    finally:
        close_logging()


def test_nested_context_is_restored_after_exception(tmp_path, capsys):
    logger = logging.getLogger("cxl_recsys.worker.test")
    try:
        path = configure_logging("worker", tmp_path, "DEBUG", 4096, 2)
        with request_context("outer", "u1"):
            with pytest.raises(RuntimeError, match="test failure"):
                with request_context("inner", "u2"):
                    logger.debug("inner_request")
                    raise RuntimeError("test failure")
            logger.debug("outer_request")
        logger.debug("outside_request")

        for output in (path.read_text(), capsys.readouterr().err):
            assert "request_id='inner' user_id='u2' inner_request" in output
            assert "request_id='outer' user_id='u1' outer_request" in output
            outside = next(
                line for line in output.splitlines() if "outside_request" in line
            )
            assert "request_id=" not in outside
            assert "user_id=" not in outside
    finally:
        close_logging()


@pytest.mark.parametrize("request_id,user_id", [("-", "-"), ("r\n1", "u\r\n2")])
def test_bound_ids_are_preserved_and_escaped(tmp_path, capsys, request_id, user_id):
    logger = logging.getLogger("cxl_recsys.worker.test")
    try:
        path = configure_logging("worker", tmp_path, "DEBUG", 4096, 2)
        with request_context(request_id, user_id):
            logger.debug("bound_request")

        for output in (path.read_text(), capsys.readouterr().err):
            assert len(output.splitlines()) == 2
            assert (
                f"request_id={request_id!r} user_id={user_id!r} bound_request" in output
            )
    finally:
        close_logging()


def test_rotation_and_exception(tmp_path):
    logger = logging.getLogger("cxl_recsys.mempool.test")
    try:
        path = configure_logging("mempool", tmp_path, "DEBUG", 1024, 2)
        for number in range(40):
            logger.debug("page_read number=%d", number)
        assert path.with_name(path.name + ".1").exists()
        assert len(list(tmp_path.glob("*.log*"))) == 3
        try:
            raise RuntimeError("test failure")
        except RuntimeError:
            logger.exception("transfer_failed")
        assert "RuntimeError: test failure" in path.read_text()
    finally:
        close_logging()


@pytest.mark.parametrize("max_bytes,backups", [(0, 2), (1024, 0)])
def test_invalid_rotation_limits(tmp_path, max_bytes, backups):
    with pytest.raises(ValueError):
        configure_logging("scheduler", tmp_path, "DEBUG", max_bytes, backups)


def legacy_log(directory, role, number, suffix=""):
    host = f"host-{number % 2 + 1}.example.org"
    path = directory / (
        f"{role}-{host}-123-20260101-0000{number:02d}-{number:08x}.log{suffix}"
    )
    path.write_text("old runtime log\n")
    return path


@pytest.mark.parametrize("role", ["scheduler", "worker", "mempool"])
def test_new_run_keeps_five_per_role_and_removes_old_backups(tmp_path, role):
    old = [legacy_log(tmp_path, role, number) for number in range(7)]
    backups = [legacy_log(tmp_path, role, number, ".1") for number in range(7)]
    # Updating the oldest file must not change its startup ordering.
    os.utime(old[0], (4_000_000_000, 4_000_000_000))
    other_role = "worker" if role == "scheduler" else "scheduler"
    unrelated = [legacy_log(tmp_path, other_role, number) for number in range(7)]
    try:
        current = configure_logging(role, tmp_path, "DEBUG", 4096, 2)
        assert current.exists()
        assert len(list(tmp_path.glob(f"{role}-*.log"))) == 5
        assert not any(path.exists() for path in old[:3] + backups[:3])
        assert all(path.exists() for path in old[3:] + backups[3:] + unrelated)
    finally:
        close_logging()


def test_cleanup_recognizes_orphan_backups_but_ignores_other_files(tmp_path):
    orphan = legacy_log(tmp_path, "worker", 0, ".2")
    kept = [legacy_log(tmp_path, "worker", number) for number in range(1, 6)]
    unrelated = tmp_path / "worker-not-a-runtime-log.log"
    unrelated.write_text("keep this file")
    target = tmp_path / "notes.txt"
    target.write_text("do not touch")
    symlink = tmp_path / "worker-host-123-20200101-000000-00000000.log"
    symlink.symlink_to(target)
    directory = tmp_path / "worker-host-123-20200101-000000-00000001.log"
    directory.mkdir()
    child = directory / "child.txt"
    child.write_text("not a log")

    assert prune_logs("worker", tmp_path) == [orphan]
    assert not orphan.exists()
    assert all(path.exists() for path in kept)
    assert unrelated.read_text() == "keep this file"
    assert symlink.is_symlink() and target.read_text() == "do not touch"
    assert child.read_text() == "not a log"
    assert prune_logs("worker", tmp_path) == []


def test_quick_repeated_startups_keep_exactly_the_newest_five(tmp_path):
    try:
        paths = [
            configure_logging("worker", tmp_path, "DEBUG", 4096, 2) for _ in range(9)
        ]
        assert len(set(paths)) == 9
        assert set(tmp_path.glob("worker-*.log")) == set(paths[-5:])
    finally:
        close_logging()


def test_concurrent_process_startups_share_retention_lock(tmp_path):
    script = (
        "import sys; "
        "from cxl_recsys.common.logging import configure_logging, close_logging; "
        "configure_logging('worker', sys.argv[1], 'DEBUG', 4096, 2); "
        "close_logging()"
    )

    def start(_):
        return subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(start, range(12)))
    assert all(result.returncode == 0 for result in results), [
        result.stderr for result in results if result.returncode
    ]
    assert len(list(tmp_path.glob("worker-*.log"))) == 5
    assert prune_logs("worker", tmp_path) == []


def test_evicted_live_run_does_not_reappear_on_rotation(tmp_path):
    logger = logging.getLogger("cxl_recsys")
    first = configure_logging("worker", tmp_path, "DEBUG", 1024, 2)
    handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    # Keep one old handler open, as if it belonged to another still-live Worker.
    logger.removeHandler(handler)
    try:
        for _ in range(5):
            configure_logging("worker", tmp_path, "DEBUG", 1024, 2)
        assert not first.exists()
        record = logging.LogRecord(
            "cxl_recsys", logging.DEBUG, "", 0, "x" * 2048, (), None
        )
        handler.handle(record)
        handler.handle(record)
        assert not list(tmp_path.glob(first.name + "*"))
        assert len(list(tmp_path.glob("worker-*.log"))) == 5
        assert handler.stream is None
    finally:
        handler.close()
        close_logging()


def test_retention_rejects_unknown_role_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match="Unknown logging role"):
        configure_logging("../worker", tmp_path, "DEBUG", 1024, 2)
    with pytest.raises(ValueError, match="Unknown logging role"):
        prune_logs("unknown", tmp_path)
    assert list(tmp_path.iterdir()) == []
