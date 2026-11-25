"""Run Wave 42 eager CUDA screening and emit a checksummed result archive."""

from __future__ import annotations

# Stdout is the job's artifact transport.
# ruff: noqa: T201
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
SCAN_OPS = ("vscan_bwd", "vscan_fwd", "hscan_fwd", "hscan_bwd")
READER_OUTPUTS = {"fwd": "out", "dinput": "d_proj", "dweight": "d_w"}
CRITERIA = ("exact_match", "tol_pass")
BENCHES = {
    "scan": ("--all-ops",),
    "reader": ("--modes", "off,ro,fwd_hiocc,dw_hiocc,dinput_tune,combo"),
}


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def device() -> dict:
    import torch  # noqa: PLC0415 - preserve transport when Torch cannot import

    gpu = torch.cuda.get_device_properties(0)
    return {
        "name": gpu.name,
        "sms": gpu.multi_processor_count,
        "vram_bytes": gpu.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": version("triton"),
    }


def select_winners(times: dict, passes: dict, baseline: str) -> dict:
    winners = {}
    for criterion in CRITERIA:
        mode = min(
            (m for m in times if passes[criterion].get(m)), key=times.__getitem__, default=None
        )
        if mode is not None:
            ms = times[mode]
            winners[criterion] = {
                "mode": mode,
                "weighted_ms": ms,
                "delta_vs_off_ms": ms - times["off"],
                "gain_vs_current_ms": times[baseline] - ms,
            }
    return winners


def scan_winners(report: dict) -> dict:
    winners = {}
    for op in SCAN_OPS:
        record = report[op]
        coverage = record["timing_kernel_covered"]
        passes = {
            criterion: {
                mode: passed and coverage["off"] and coverage[mode]
                for mode, passed in record[key].items()
            }
            for criterion, key in zip(CRITERIA, ("parity_pass", "tolerance_pass"), strict=True)
        }
        winners[op] = select_winners(
            record["weighted_ms"], passes, "strict" if op == "vscan_bwd" else "off"
        )
    return winners


def reader_winners(report: dict) -> dict:
    winners = {}
    for op, output in READER_OUTPUTS.items():
        times = {mode: values[op] for mode, values in report["weighted_ms"].items()}
        passes = {
            criterion: {
                mode: bool(dtypes)
                and all(
                    shape[output][criterion]
                    for shapes in dtypes.values()
                    for shape in shapes.values()
                )
                for mode, dtypes in report["parity"].items()
            }
            for criterion in CRITERIA
        }
        winners[f"reader_{op}"] = select_winners(times, passes, "off")
    return winners


def emit(root: Path) -> None:
    names = ("device.json", "scan_micro.json", "reader_micro.json", "summary.json")
    manifest = root / "manifest.json"
    write(manifest, {name: digest((root / name).read_bytes()) for name in names})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as archive:
        for name in (*names, manifest.name):
            archive.add(root / name, arcname=name)
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"VA_T148_SHA256 {digest(raw)}")
    print(f"VA_T148_MANIFEST_SHA256 {digest(manifest.read_bytes())}")
    print(f"VA_T148_BASE64_CHARS {len(encoded)}")
    print("VA_T148_BEGIN")
    for start in range(0, len(encoded), 480):
        print(f"VA_T148_GZ {encoded[start : start + 480]}")
    print("VA_T148_END", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True)
    started = perf_counter()
    os.environ["CUDA_CACHE_PATH"] = "/tmp/va_w42_cuda_cache"  # noqa: S108
    for key in tuple(os.environ):
        if key.startswith("LNET_"):
            del os.environ[key]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=ROOT,
        text=True,
    ).strip()
    summary = {"commit": commit, "winners": {}}
    try:
        write(out / "device.json", device())
    except (RuntimeError, AssertionError, ImportError, OSError, ValueError) as exc:
        write(out / "device.json", {"error": str(exc)})
        summary["device_error"] = str(exc)
    for name, flags in BENCHES.items():
        path = out / f"{name}_micro.json"
        step = perf_counter()
        rc = 1
        try:
            rc = subprocess.run(  # noqa: S603 - fixed local scripts
                [
                    sys.executable,
                    f"scripts/bench_{name}_micro.py",
                    *flags,
                    "--batch",
                    "1024",
                    "--dtype",
                    "bf16",
                    "--warmup",
                    "5",
                    "--iters",
                    "20",
                    "--parity",
                    "--out",
                    str(path),
                ],
                cwd=ROOT,
                check=True,
                timeout=600,
            ).returncode
            summarize = scan_winners if name == "scan" else reader_winners
            summary["winners"].update(summarize(json.loads(path.read_text(encoding="utf-8"))))
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                rc = exc.returncode
            elif isinstance(exc, subprocess.TimeoutExpired):
                rc = 124
            summary[f"{name}_error"] = str(exc)
            if not path.exists():
                write(path, {"error": str(exc)})
        summary[f"{name}_wall_s"] = perf_counter() - step
        summary[f"{name}_exit_code"] = rc
        print(f"{name} rc={rc} wall_s={summary[f'{name}_wall_s']:.3f}", flush=True)
    for criterion in CRITERIA:
        selected = [
            choices[criterion] for choices in summary["winners"].values() if criterion in choices
        ]
        gain = sum(choice["gain_vs_current_ms"] for choice in selected)
        summary[f"{criterion}_rollup"] = {
            "ops": len(selected),
            "gain_vs_off_ms": -sum(choice["delta_vs_off_ms"] for choice in selected),
            "gain_vs_current_ms": gain,
            "projected_ms_from_wave40_182_46": 182.46 - gain if len(selected) == 7 else None,
        }
    summary["total_wall_s"] = perf_counter() - started
    write(out / "summary.json", summary)
    emit(out)
    return int(any(key.endswith("_error") for key in summary))


if __name__ == "__main__":
    raise SystemExit(main())
