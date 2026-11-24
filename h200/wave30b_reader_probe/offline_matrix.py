"""Replay Wave 30B production launches and assemble their NVRTC PTX offline."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, SLF001
import argparse
import hashlib
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import replay
import torch
from screen import GENERAL, REAL

from lnet import factorized_reader_kernels as reader

ARMS = ("off", "auto", *reader._WAVE30B_PRESETS)
LEGACY = ("auto", *reader._WAVE29B_PRESETS)


@cache
def load_reference(path, wave):
    spec = spec_from_file_location(f"lnet._wave{wave}_reference", path)
    module = module_from_spec(spec)
    source = path.read_text().replace('"va_reader::', f'"va_reader_wave{wave}::')
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def audit(job):  # noqa: C901
    sm, shape, ref26, ref29, ptxas = job
    reference26, reference29 = load_reference(ref26, 26), load_reference(ref29, 29)
    cases = [
        (shape, (dtype,) * 3, strict)
        for dtype, strict in itertools.product((torch.bfloat16, torch.float16), (False, True))
    ]
    original_resources = replay.resources
    preset = reader._wave30b_preset
    previous = {}

    def run(module, w25, w29, arm):
        os.environ.update(LNET_READER_W25B=w25, LNET_READER_W29B=w29, LNET_READER_W30B=arm)
        with TemporaryDirectory(prefix="va-w30b-") as scratch:
            directory = Path(scratch)

            def resources(log):
                digest = hashlib.sha256((directory / "reader.ptx").read_bytes()).hexdigest()
                return {
                    key: {**data, "ptx_sha256": digest}
                    for key, data in original_resources(log).items()
                }

            def force_preset(*args, **kwargs):
                with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                    return preset(*args, **kwargs)

            force = arm in reader._WAVE30B_PRESETS and sm != "90"
            with (
                patch.object(module._reader_cuda_module, "cache_clear", lambda: None),
                patch.object(replay, "resources", resources),
                patch.object(reader, "_wave30b_preset", force_preset if force else preset),
            ):
                return list(replay.replay(module, sm, cases, directory, ptxas))

    for module in (reference26, reference29, reader):
        module._reader_cuda_module.cache_clear()
    records = []
    for w25 in ("auto", "off"):
        baseline = run(reference26, w25, "off", "off")
        for arm in ARMS:
            current = run(reader, w25, "off", arm)
            previous[arm] = current
            records.extend({"arm": arm, "w25b": w25, "w29b": "off", **row} for row in current)
        if baseline != previous["off"]:
            raise RuntimeError(f"Wave 26/quad-off mismatch: {sm} {shape} {w25}")
        expected = "full_w30b" if sm == "90" else "off"
        if previous[expected] != previous["auto"]:
            raise RuntimeError(f"Auto dispatch mismatch: {sm} {shape} {w25}")
        for row, full in enumerate(previous["full_w30b"]):
            for op, variant in {
                "forward": "wide_col_spatial", "dinput": "wide_col_spatial",
                "dweight": "dweight_low_traffic", "dweight_reduce": "dweight_low_traffic",
                "dweight_reduce_parallel": "dweight_low_traffic",
            }.items():
                actual = full["dispatch"].get(op, {})
                expected_op = previous[variant][row]["dispatch"].get(op, {})
                # Compact dinput can share a module with a different dweight kernel.
                if {k: v for k, v in actual.items() if k != "ptx_sha256"} != {
                    k: v for k, v in expected_op.items() if k != "ptx_sha256"
                }:
                    raise RuntimeError(f"Composition mismatch: {sm} {shape} {w25} {op}")
        if sm == "90":
            winner = run(reference29, w25, "hiocc_dinput_v2", "off")
            if winner != run(reader, w25, "hiocc_dinput_v2", "off"):
                raise RuntimeError(f"Wave 29B winner mismatch: {sm} {shape} {w25}")
            records.extend({"arm": "base", "w25b": w25, "w29b": "hiocc_dinput_v2", **row}
                           for row in winner)
        for w29 in LEGACY:
            base = run(reference29, w25, w29, "off")
            current = run(reader, w25, w29, "off")
            if current != base:
                raise RuntimeError(f"Wave 29B compatibility mismatch: {sm} {shape} {w25} {w29}")
            records.extend({"arm": "off", "w25b": w25, "w29b": w29, **row} for row in current)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref26", type=Path, required=True)
    parser.add_argument("--ref29", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ptxas", default="/usr/local/cuda-12.8/bin/ptxas")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    os.environ.update(LNET_READER_W27B="off", LNET_READER_W28B="off", LNET_READER_VEC="auto")
    jobs = (
        (sm, (batch, *shape), args.ref26, args.ref29, args.ptxas)
        for sm, batch, shape in itertools.product(replay.SM, (1024, 128), (*REAL, *GENERAL))
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
