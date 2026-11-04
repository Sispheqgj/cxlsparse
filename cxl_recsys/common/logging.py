"""Process-local application logging; never configure third-party loggers."""

import fcntl
import logging
import os
import re
import socket
import stat
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import PROJECT_ROOT

_request: ContextVar[tuple[str, str] | None] = ContextVar("log_request", default=None)

LOG_ROLES = ("scheduler", "worker", "mempool")
RETAINED_RUNS = 5
_RUN_NAME = re.compile(
    r"(?P<run>(?P<role>scheduler|worker|mempool)-[a-zA-Z0-9_.-]+-\d+-"
    r"(?P<stamp>\d{8}-\d{6}(?:\d{6})?)-[0-9a-f]{8}\.log)"
    r"(?:\.[1-9]\d*)?"
)


def _log_directory(directory):
    directory = Path(directory)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return directory


def _validate_role(role):
    if role not in LOG_ROLES:
        raise ValueError(f"Unknown logging role: {role}")


@contextmanager
def _retention_lock(directory, role):
    # Persistent lock files must not be removed: another process may already
    # hold the inode. flock also coordinates hosts on our NFSv4 shared mount.
    with (directory / f".{role}-retention.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _prune_runs(directory, role):
    """Caller holds the role lock, including against file rotation."""
    groups = {}
    order = {}
    for path in directory.iterdir():
        match = _RUN_NAME.fullmatch(path.name)
        if match is None or match["role"] != role:
            continue
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                continue  # Never follow symlinks or recursively remove a directory.
        except FileNotFoundError:
            continue
        run = match["run"]
        groups.setdefault(run, []).append(path)
        # Sort by startup time, not mtime: writing/rolling an old run must not
        # make it newer. Legacy timestamps lack the six microsecond digits.
        order[run] = (match["stamp"].ljust(21, "0"), run)

    newest_first = sorted(groups, key=order.__getitem__, reverse=True)
    removed = []
    for run in newest_first[RETAINED_RUNS:]:
        for path in groups[run]:
            try:
                path.unlink()
                removed.append(path)
            except FileNotFoundError:
                pass
    return removed


def prune_logs(role, directory):
    """Remove files from all but the newest five runs of one role.

    A run includes its .log and all numeric rotation backups. Host names do not
    split the retention quota. Only recognized regular log files are eligible.
    """
    _validate_role(role)
    directory = _log_directory(directory)
    if not directory.exists():
        return []
    with _retention_lock(directory, role):
        return _prune_runs(directory, role)


class RetainedRunFileHandler(RotatingFileHandler):
    """Serialize rotation with retention; never recreate an evicted old run."""

    def __init__(self, role, path, max_bytes, backup_count):
        self.role = role
        self.retired = False
        super().__init__(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )

    def emit(self, record):
        if self.retired:
            return
        try:
            if self.shouldRollover(record):
                path = Path(self.baseFilename)
                with _retention_lock(path.parent, self.role):
                    if not path.exists():
                        # More than five newer runs may start while this process
                        # is still alive. Its console handler remains unaffected.
                        self.retired = True
                        if self.stream is not None:
                            self.stream.close()
                            self.stream = None
                        return
                    self.doRollover()
                    logging.FileHandler.emit(self, record)
            else:
                logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)


@contextmanager
def request_context(request_id, user_id):
    token = _request.set((request_id, user_id))
    try:
        yield
    finally:
        _request.reset(token)


class RuntimeFilter(logging.Filter):
    def filter(self, record):
        record.host = socket.gethostname()
        context = _request.get()
        record.request_context = ""
        if context is not None:
            request_id, user_id = context
            # repr keeps user-controlled identifiers on one log line.
            record.request_context = f" request_id={request_id!r} user_id={user_id!r}"
        return True


def close_logging():
    logger = logging.getLogger("cxl_recsys")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(role, directory, level, max_bytes, backup_count):
    _validate_role(role)
    if max_bytes <= 0 or backup_count < 1:
        raise ValueError("log_max_bytes must be positive and log_backup_count >= 1")
    directory = _log_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with _retention_lock(directory, role):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        path = directory / (
            f"{role}-{socket.gethostname()}-{os.getpid()}-{stamp}-{uuid.uuid4().hex[:8]}.log"
        )
        file_handler = RetainedRunFileHandler(role, path, max_bytes, backup_count)
        try:
            removed = _prune_runs(directory, role)
        except BaseException:
            file_handler.close()
            raise
    close_logging()
    logger = logging.getLogger("cxl_recsys")
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s host=%(host)s pid=%(process)d "
        "thread=%(threadName)s%(request_context)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for handler in (logging.StreamHandler(), file_handler):
        handler.setFormatter(formatter)
        handler.addFilter(RuntimeFilter())
        logger.addHandler(handler)
    logger.info("logging_started role=%s level=%s file=%s", role, level, path)
    if removed:
        logger.info(
            "log_retention role=%s retained_runs=%d removed_files=%d",
            role,
            RETAINED_RUNS,
            len(removed),
        )
    return path
