#!/usr/bin/env python3
"""Wait for prerequisite queue receipts, then replace this process with a command."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

TERMINAL_STATES = {"COMPLETE", "FAILED"}


def _complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        jobs = payload["jobs"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return bool(jobs) and all(job.get("status") in TERMINAL_STATES for job in jobs.values())


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, action="append", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0.0 or not args.command:
        raise ValueError("queue chain requires a positive poll interval and command")
    if args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("queue chain command cannot be empty")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    while not all(_complete(path) for path in args.status):
        time.sleep(args.poll_seconds)
    os.execvpe(args.command[0], args.command, os.environ.copy())  # noqa: S606


if __name__ == "__main__":
    main()
