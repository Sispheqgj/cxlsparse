"""Workload loading and scheduling; no Worker/GPU dependency at import time."""

from .service import Scheduler

__all__ = ["Scheduler"]
