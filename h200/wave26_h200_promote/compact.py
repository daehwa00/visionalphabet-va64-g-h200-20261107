"""Preserve timings and dispatch when full traces exceed transport capacity."""

import json
import sys
from pathlib import Path


def compact(directory: Path) -> None:
    for path in directory.glob("b*.json"):
        row = json.loads(path.read_text())
        if not (profile := row.get("profiler")):
            continue
        if kernels := profile.get("kernels"):
            profile["kernels"] = [
                k for k in kernels
                if k["name"].startswith((
                    "reader_", "stem_conv_tensor", "va_t93_step",
                    "## Call CompiledFxGraph", "Optimizer.step#",
                ))
            ]
        profile.pop("top_kernels", None)
        profile["transport_compacted"] = True
        path.write_text(json.dumps(row, separators=(",", ":")))
    for path in directory.glob("*.log"):
        if path.name not in {"cuda-tests.log", "original-unroll.log"}:
            path.unlink()


if __name__ == "__main__":
    compact(Path(sys.argv[1]))
