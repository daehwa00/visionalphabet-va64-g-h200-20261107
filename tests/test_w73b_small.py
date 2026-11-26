# ruff: noqa: ANN201, S607
"""Compile the real W57 launch presets and the corresponding small FP8 entries offline."""

import concurrent.futures
import re
import subprocess

import torch
from test_w66_split_bf16 import compile_source, offline_plans

from lnet import w73b_small_fp8_scan as fp

SHAPES = (("14d", 14, 14, 160, False), ("14c", 14, 14, 160, True), ("7d", 7, 7, 128, False))


def test_words():
    for _, h, w, m, coarse in SHAPES:
        real = torch.empty((1, h, w, m))
        for width in (w, w // (1 + coarse)):
            words = fp._words(real, width)
            fp._check_storage(real, words, width)
            assert words.dtype == torch.int16
            assert words.numel() == 2 * h * width * m


def test_sm90a(tmp_path):
    def audit(job):
        label, op, w, m, launch, candidate = job
        folder = tmp_path / f"{label}-{op}-{candidate}"
        folder.mkdir()
        if candidate and op == "VB":
            launch = fp.configure(launch, w)
        if candidate and op == "HF":
            launch = fp.cached_horizontal(launch, w)
        k = launch.key
        if candidate:
            src = fp.kernel_source(k, w, w, m)
            symbol = fp.profile_name(k)
        else:
            src = fp.scans._kernel_source(k)
            symbol = fp.scans._kernel_name(k)
        flags = fp.compile_flags(k, w)
        result = compile_source(src, folder, flags, symbol)
        (cu,) = folder.glob("*.cu")
        ptx = subprocess.check_output(
            ["nvcc", "-arch=sm_90a", "-std=c++17", *flags, "-ptx", str(cu), "-o", "/dev/stdout"],
            text=True,
        )
        entry = ptx.split(".entry " + symbol + "(", 1)[1].split(".entry ", 1)[0]
        formats = tuple(bool(re.search(r"cvt[^;]*\." + fmt, entry)) for fmt in ("e4m3x2", "e5m2x2"))
        if candidate:
            assert formats == (True, op in ("VB", "HB"))
            assert result["stack"] == result["LDL"] == result["STL"] == 0
            if op == "VF" and w == 14:
                assert "fma.rn.bf16x2" in entry
            if op == "VB":
                assert k.sgs
                assert k.cache_line
                assert "words[ia].unpack" not in src
                assert "write_column_gradient<true, COMPACT, false>" in src
                assert src.count("words[ia]=") == src.count("words[ib]=") == 1
        smem = (fp.shared_bytes(launch, w) if candidate else launch.plan.smem) + result["static"]
        reg = result["regs"]
        aw = k.warps * min(
            32,
            64 // k.warps,
            65536 // (k.warps * ((reg + 7) // 8) * 256),
            (228 << 10) // (((smem + 255) // 256) * 256 + 1024),
        )
        return (
            label,
            op,
            candidate,
            reg,
            smem,
            aw,
            result["stack"],
            result["LDL"],
            result["STL"],
            *formats,
        )

    jobs = [(*row, candidate) for row in offline_plans(SHAPES) for candidate in (False, True)]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for row in pool.map(audit, jobs):
            print(*row)  # noqa: T201
