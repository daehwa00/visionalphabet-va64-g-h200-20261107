"""Audit Wave 28B production dispatch through NVRTC and CUDA 12.8 ptxas."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201, SLF001
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

ARMS = ("base", "off", "auto", *reader._WAVE28B_PRESETS)


def audit(job):
    sm, shape, baseline, ptxas = job
    reference = load_baseline(baseline)
    cases = [
        (shape, (dtype,) * 3, strict)
        for dtype, strict in itertools.product((torch.bfloat16, torch.float16), (False, True))
    ]
    original_preset = reader._wave28b_preset
    original_resources = replay.resources
    records = {}
    for w25b in ("auto", "off"):
        os.environ["LNET_READER_W25B"] = w25b
        for arm in ARMS:
            os.environ["LNET_READER_W28B"] = "off" if arm == "base" else arm
            with TemporaryDirectory(prefix="va-w28b-") as scratch:
                directory = Path(scratch)

                def resources(log: str, directory: Path = directory) -> dict:
                    kernels = original_resources(log)
                    ptx_hash = hashlib.sha256((directory / "reader.ptx").read_bytes()).hexdigest()
                    for kernel in kernels.values():
                        kernel["ptx_sha256"] = ptx_hash
                    return kernels

                def force_preset(*args, **kwargs) -> str:
                    # Open explicit candidates for offline SM89 assembly only.
                    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                        return original_preset(*args, **kwargs)

                forced = sm == "89" and arm in reader._WAVE28B_PRESETS
                gate = force_preset if forced else original_preset
                with (
                    patch.object(replay, "resources", resources),
                    patch.object(reader, "_wave28b_preset", gate),
                ):
                    module = reference if arm == "base" else reader
                    records[w25b, arm] = list(replay.replay(module, sm, cases, directory, ptxas))
        if records[w25b, "base"] != records[w25b, "off"]:
            raise RuntimeError(f"Wave 26/off dispatch, PTX or SASS mismatch: {sm} {shape} {w25b}")
        expected_auto = "keep_fwd_hopper" if sm == "90" else "off"
        if records[w25b, expected_auto] != records[w25b, "auto"]:
            raise RuntimeError(f"Unexpected auto dispatch: {sm} {shape} {w25b}")
    return [
        {"arm": arm, "w25b": w25b, **row}
        for (w25b, arm), results in records.items()
        for row in results
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ptxas", default="/usr/local/cuda-12.8/bin/ptxas")
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    os.environ.update(LNET_READER_W27B="off", LNET_READER_VEC="auto")
    jobs = (
        (sm, (batch, *shape), args.baseline, args.ptxas)
        for sm, batch, shape in itertools.product(replay.SM, (128, 1024), (*REAL, *GENERAL))
    )
    failed = False
    with args.out.open("w") as out, ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for records in pool.map(audit, jobs):
            for record in records:
                print(json.dumps(record), file=out, flush=True)
                failed |= any(any(kernel["stack_spill"]) for kernel in record["dispatch"].values())
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
