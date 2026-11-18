#!/usr/bin/env python3
"""Resume the CLPost confirmation with a math-equivalent packed runtime."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as packed

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = packed.VARIANT
VARIANTS = packed.VARIANTS
SEEDS = packed.SEEDS
_build = packed._build
_assert_model = packed._assert_model


def _contract(args: Namespace) -> dict[str, Any]:
    existing = Path(args.root) / "contract.json"
    if not existing.exists():
        return packed._contract(args)
    payload = json.loads(existing.read_text())
    if payload.get("variants") != list(VARIANTS):
        raise RuntimeError("packed resume found an incompatible existing variant contract")
    return payload


def main() -> None:
    runtime.run(
        variants=VARIANTS,
        seeds=SEEDS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
