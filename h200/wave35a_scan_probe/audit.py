"""Offline Wave 35A PTX/SASS audit, with no CUDA device context."""
# ruff: noqa: ANN001, ANN201, ANN202, E501, PT018, S101, S102, S603, S607, SLF001, T201

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import torch

from lnet import general_cuda_scans as scans

ARCHES = {(8, 6): (100, 48), (8, 9): (100, 48), (9, 0): (228, 64), (10, 0): (228, 64)}
SHAPES = [
    (1024, 56, 56, 160, False), (1024, 56, 56, 160, True),
    (1024, 28, 28, 128, False), (1024, 14, 14, 64, False),
    (2, 13, 19, 63, False), (2, 14, 20, 63, True), (2, 7, 7, 63, False),
    (2, 37, 13, 63, False),
]


def load_base():
    name = "lnet._wave35a_audit_base"
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp, "base.py")
        file.write_bytes(subprocess.check_output([
            "git", "show", "235c026b:src/lnet/general_cuda_scans.py",
        ]))
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Avoid registering the same torch.library definitions twice.
        exec(compile(file.read_text().split("# Custom ops and public API")[0], str(file), "exec"),
             module.__dict__)
        return module


def stub_device(module, smem, max_warps):
    module.device_limits = lambda _: (132, smem * 1024 - 1024, max_warps)
    module._scan_workspace_budget = lambda _: 64 * 1024**2
    module._horizontal_cache_budget = lambda _: 12 * 1024**2
    module._l2_size = lambda _: 50 * 1024**2
    # Deterministic occupancy fixture for baseline planner equivalence.
    module._resident = lambda _, key, dynamic: min(
        max_warps // key.warps, smem * 1024 // max(1, dynamic), 8,
    )


@cache
def inspect_binary(ptx, arch):
    with tempfile.TemporaryDirectory() as tmp:
        asm, cubin = Path(tmp, "scan.ptx"), Path(tmp, "scan.cubin")
        asm.write_bytes(ptx)
        result = subprocess.run(
            ["ptxas", "-v", f"-arch=sm_{arch}", asm, "-o", cubin],
            check=True, capture_output=True, text=True,
        ).stderr
        sass = subprocess.check_output(["nvdisasm", "-ndf", cubin], text=True)
    registers = int(re.search(r"Used (\d+) registers", result)[1])
    stack, spills, loads = map(int, re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", result,
    ).groups())
    counts = Counter(re.findall(r"\b(LDS|STS|LDG|STG|FFMA)\b", sass))
    return registers, stack, spills, loads, *(counts[k] for k in (
        "LDS", "STS", "LDG", "STG", "FFMA",
    ))


def inspect(key, ptx, arch, smem, capacity, max_warps):
    registers, *metrics = inspect_binary(ptx, arch)
    allocation = ((registers + 7) // 8) * 256
    warp_limit = min(max_warps, 4 * (16384 // allocation))
    block_limit = {"86": 16, "89": 24, "90": 32, "100": 32}[arch]
    ctas = min(warp_limit // key.warps, capacity // (smem + 1024), block_limit)
    return registers, smem, ctas * key.warps, *metrics


def base_launch(module, source, shape, vertical, backward, coarse):
    return module._plan_launch(
        source, shape, backward=backward, vertical=vertical, coarse=coarse if vertical else False,
        hidden=8 if vertical else 0, precise=False, compact=coarse,
    )


def main():  # noqa: C901, PLR0912, PLR0915
    os.environ["LNET_SCAN_W35A"] = "off"
    os.environ["LNET_GENERAL_SCAN_OPT"] = "auto"
    base = load_base()
    cache = {}
    metrics = {}
    variants = {}
    checked = 0
    print("SM dtype shape kernel regs smem warps stack spillStore spillLoad LDS STS LDG STG FFMA")

    def emit(label, key, smem):
        item = cap, key, smem
        if item not in metrics:
            ptx = cache.get((cap, key))
            if ptx is None:
                if (cap, key) not in variants:
                    variants[cap, key] = scans._compile_ptx(key, cap).value
                ptx = variants[cap, key]
            metrics[item] = inspect(key, ptx, arch, smem, capacity * 1024, max_warps)
        print(arch, str(dtype).split(".")[-1], f"{h}x{w}/{m}/{int(coarse)}", label, *metrics[item])

    fields = set(base.Kernel.__dataclass_fields__)
    for cap, (capacity, max_warps) in ARCHES.items():
        arch = "".join(map(str, cap))
        for module in (base, scans):
            stub_device(module, capacity, max_warps)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for b, h, w, m, coarse in SHAPES:
                shape = b, h, w, m
                for vertical in (False, True):
                    source = SimpleNamespace(
                        dtype=dtype, is_cuda=True, device=SimpleNamespace(index=0),
                        numel=lambda n=b * h * w * m * (4 if vertical else 1): n,
                    )
                    for backward in (False, True):
                        new = base_launch(scans, source, shape, vertical, backward, coarse)
                        old = base_launch(base, source, shape, vertical, backward, coarse)
                        assert new.plan == old.plan and new[2:] == old[2:]
                        key = {k: v for k, v in asdict(new.key).items() if k in fields}
                        assert key == asdict(old.key)
                        item = cap, new.key
                        if item not in cache:
                            ptx = cache[item] = scans._compile_ptx(new.key, cap).value
                            original = base._compile_ptx(old.key, cap).value
                            assert hashlib.sha256(ptx).digest() == hashlib.sha256(original).digest()
                        checked += 1
                        if dtype != torch.float32:
                            label = f"w33_{'v' if vertical else 'h'}_{'bwd' if backward else 'fwd'}"
                            emit(label, new.key, new.plan.smem)
                if dtype == torch.float32:
                    continue

                os.environ["LNET_SCAN_W35A"] = "vscan_coop_bwd"
                source = SimpleNamespace(
                    dtype=dtype, is_cuda=True, device=SimpleNamespace(index=0),
                    numel=lambda n=4 * b * h * w * m: n, element_size=lambda: 2,
                )
                coop = scans._plan_vertical_cooperative(
                    source, shape, coarse=coarse, compact=coarse, interleaved=(False, False),
                )
                if coop is not None:
                    emit("coop_bwd", coop.key, coop.plan.smem)
                else:
                    key = scans.Kernel(dtype, 4, backward=True, vertical=True, coarse=coarse,
                                       compact=coarse, hidden8=True, cooperative=True)
                    size = 128 * (16 * (8 + (h - 1) // 8) + 8 * (h // (1 + coarse)) + 148)
                    emit("coop_bwd_ineligible", key, size)
                if not m % 2:
                    for backward in (False, True):
                        key = scans.Kernel(dtype, 2 if backward else 4, backward, compact=coarse,
                                           horizontal_vec2=True)
                        emit("h_vec2_bwd" if backward else "h_vec2_fwd", key, 16384 if backward else 0)
                for backward in (False, True):
                    key = scans.Kernel(dtype, 4, backward, projection=True, hidden8=True, aosoa=True)
                    emit("aosoa_project_bwd" if backward else "aosoa_project_fwd",
                         key, scans._AOSOA_PROJECTION_SMEM)
                    feature_bound = b * (h // (1 + coarse)) * (w // (1 + coarse))
                    feature_bound *= ((m + 31) // 32) * 8 * 32
                    key = scans.Kernel(dtype, 2, backward, vertical=True, pure_vertical=True,
                                       coarse=coarse, compact=coarse, aosoa=True, cache_line=True,
                                       index32=max(8 * b * h * w * m, feature_bound) < 2**31)
                    oh = h // (1 + coarse)
                    size = 64 * (oh * 8 + ((8 * 24 + ((h - 1) // 8) * 16) if backward else 0))
                    emit("aosoa_pure_bwd" if backward else "aosoa_pure_fwd", key, size)
                    if backward:
                        key = scans.replace(key, single_store=True)
                        emit("aosoa_pure_bwd_single_store", key, size + h * 64 * 8)
                os.environ["LNET_SCAN_W35A"] = "off"
    print(f"Off planner cases matched: {checked}; unique architecture/PTX pairs: {len(cache)}")


if __name__ == "__main__":
    main()
