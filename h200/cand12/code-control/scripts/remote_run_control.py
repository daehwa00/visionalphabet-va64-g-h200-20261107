"""Shared owner-control contract for managed long-running jobs."""

from __future__ import annotations

from typing import Any

SCHEMA = "lnet.remote_run_control.v1"


class StopRequestedError(RuntimeError):
    """Raised at a safe polling boundary after a valid stop becomes latched."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("repository owner requested a graceful stop")
        self.record = record


__all__ = ["SCHEMA", "StopRequestedError"]
