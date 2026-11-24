"""Replay live dispatch with CUDA 12.8 ptxas and compare baseline PTX and SASS."""

# ruff: noqa: ANN001, ANN201, SLF001, T201
import argparse
import hashlib
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import replay
import torch
from bench import load_baseline
from screen import GENERAL, REAL

from lnet import factorized_reader_kernels as reader

ARMS = ("base", "off", "dweight_2cta", "scalar_hiocc", "full_hopper", "auto")


def run(job):
    sm, shape, w25b, baseline, ptxas = job
    module = load_baseline(baseline)
    cases = [
        (shape, (dtype,) * 3, strict)
        for dtype, strict in itertools.product((torch.bfloat16, torch.float16), (False, True))
    ]
    os.environ["LNET_READER_W25B"] = w25b
    records = []
    original_preset = reader._wave27b_preset
    original_resources = replay.resources
    for arm in ARMS if w25b == "auto" else ARMS[:2]:
        os.environ["LNET_READER_W27B"] = "off" if arm == "base" else arm
        with TemporaryDirectory(prefix="va-w27b-") as scratch:
            directory = Path(scratch)
            compiled = []

            def resources(log, directory: Path = directory, compiled: list = compiled) -> dict:
                kernels = original_resources(log)
                ptx_hash = hashlib.sha256((directory / "reader.ptx").read_bytes()).hexdigest()
                for kernel in kernels.values():
                    kernel["ptx_sha256"] = ptx_hash
                compiled.append(kernels)
                return kernels

            def force_preset(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
                # Exercise explicit candidates at SM 8.9 while preserving the real arch
                # for NVRTC/ptxas and preserving the production auto gate.
                with patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)):
                    return original_preset(*args, **kwargs)

            gate = force_preset if sm == "89" and arm in ARMS[2:-1] else original_preset
            with (
                patch.object(replay, "resources", resources),
                patch.object(reader, "_wave27b_preset", gate),
            ):
                result = list(replay.replay(
                    module if arm == "base" else reader, sm, cases, directory, ptxas,
                ))
            records.extend({"arm": arm, "w25b": w25b, **record} for record in result)
            if arm in ARMS[2:]:
                b, h, w, _, _ = shape
                for kernels in compiled:
                    for op, kernel in kernels.items():
                        if any(kernel["stack_spill"]):
                            print(f"local memory SM{sm} {b}x{h}x{w} {arm} {op}: {kernel}")
    base = [record["dispatch"] for record in records if record["arm"] == "base"]
    off = [record["dispatch"] for record in records if record["arm"] == "off"]
    if base != off:
        raise RuntimeError(f"Off dispatch/PTX/SASS mismatch: {sm} {shape} W25B={w25b}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ptxas", default="/usr/local/cuda-12.8/bin/ptxas")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    shapes = (*REAL, *GENERAL)
    jobs = (
        (sm, (batch, *shape), w25b, args.baseline, args.ptxas)
        for sm, shape, batch, w25b in itertools.product(
            replay.SM, shapes, (128, 1024), ("auto", "off"),
        )
    )
    failed = False
    with args.out.open("w") as out, ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for records in pool.map(run, jobs):
            for record in records:
                print(json.dumps(record), file=out, flush=True)
                failed |= any(any(kernel["stack_spill"]) for kernel in record["dispatch"].values())
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
