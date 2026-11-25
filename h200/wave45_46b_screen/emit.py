"""Print the H200 screen and transport the original JSON bytes in issue stdout."""

from __future__ import annotations

# ruff: noqa: E501, FBT001, S603, S607, T201
import base64
import gzip
import json
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
import triton


def verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def scan_summary(report: dict) -> None:
    for mode, samples in report["shapes"].items():
        print(
            "scan",
            mode,
            *(f"{label}={row['median_ms']:.4f}" for label, row in samples.items()),
            f"weighted={report['weighted_ms'][mode]:.4f}",
            f"delta_strict_ro={report['delta_vs_w43_ms'][mode]:+.4f}",
            f"delta_off={report['delta_vs_off_ms'][mode]:+.4f}",
        )
    print("scan NVRTC", report["jit"], "parity_complete", report["parity_complete"])
    for mode in report["weighted_ms"]:
        for dtype in ("bf16", "fp16"):
            tensors = [
                result
                for name, case in report.get("parity", {}).items()
                if name.startswith(dtype + "/")
                for result in case["modes"][mode]["tensors"].values()
            ]
            if tensors:
                print(
                    "scan parity",
                    mode,
                    dtype,
                    *(
                        f"{key}={max((t[key] for t in tensors if t[key] is not None), default=None)}"
                        for key in ("max_abs", "max_rel")
                    ),
                    verdict(all(t["tol_pass"] for t in tensors)),
                )


def readproj_summary(report: dict) -> None:
    totals = report["weighted_model_ms"]
    for row in report["rows"]:
        if row["model_calls"]:
            print(
                "readproj",
                row["shape"],
                row["mode"],
                f"fwd={row['fwd_ms']:.4f} bwd={row['bwd_ms']:.4f}",
                f"total={row['fwd_ms'] + row['bwd_ms']:.4f}",
            )
    for mode, times in totals.items():
        tensors = [
            tensor
            for row in report["rows"]
            if row["mode"] == mode
            for tensor in row["parity"].values()
        ]
        print(
            "readproj weighted",
            mode,
            *(f"{k}={v:.4f}" for k, v in times.items()),
            f"delta_off={times['total_ms'] - totals['off']['total_ms']:+.4f}",
            "parity",
            verdict(all(t["passed"] for t in tensors)),
        )


def main(out: Path) -> None:
    stages = {}
    for line in (out / "stages.tsv").read_text().splitlines():
        stage, status, seconds = line.split("\t")
        stages[stage] = {"exit": int(status), "seconds": int(seconds)}
    device = torch.cuda.get_device_properties(0)
    meta = {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "image": "pytorch/pytorch:latest",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": device.name,
        "gpu_count": torch.cuda.device_count(),
        "capability": torch.cuda.get_device_capability(0),
        "sm_count": device.multi_processor_count,
        "vram_bytes": device.total_memory,
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "bootstrap_seconds": 0,
        "stages": stages,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    artifacts = (
        (out / "meta.json", None),
        (out / "scan_vscan_bwd.json", scan_summary),
        (out / "readproj_wave46b.json", readproj_summary),
    )
    for path, summary in artifacts:
        if not path.is_file():
            continue
        checksum = subprocess.check_output(["sha256sum", path.name], cwd=out, text=True).strip()
        print(checksum)
        print(f"VA_T162_BEGIN {path.name} {checksum.split()[0]}")
        payload = base64.b64encode(gzip.compress(path.read_bytes(), mtime=0)).decode("ascii")
        print(textwrap.fill(payload, width=120))
        print(f"VA_T162_END {path.name}", flush=True)
        if summary is not None:
            summary(json.loads(path.read_bytes()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
