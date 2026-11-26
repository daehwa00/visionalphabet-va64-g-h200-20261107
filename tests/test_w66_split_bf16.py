# ruff: noqa: ANN201, B023, C408, E501, PLC0415, PLW1510, RUF059, S607, T201
"""Offline audit of the two large shapes and Hopper-only numerical checks."""
import concurrent.futures
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_wave66_split_bf16 as bench
import pytest
import torch
from bench_wave66_split_bf16 import OPS, SHAPES, VARIANTS, fp, scans


def offline_plans():
    from contextlib import ExitStack

    with ExitStack() as stack:
        for name, fn in vars(scans).items():
            if name.startswith(("_plan_", "_launch_for_shape", "_lockin_")) and hasattr(
                fn, "cache_clear"
            ):
                fn.cache_clear()
                stack.callback(fn.cache_clear)
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    f"LNET_SCAN_{k}": v
                    for k, v in dict(
                        SGS="strict_ro",
                        W53A="w57_lockin",
                        W51A="fp32",
                        VFWD="batch4_w2",
                        HSCAN="h55_lockin",
                    ).items()
                }
                | {"LNET_GENERAL_SCAN_OPT": "auto", "LNET_VSCAN_DECOUPLED": "off"},
            )
        )
        limits = {
            "device_limits": lambda _: (132, 227 << 10, 64),
            "_sgs_device_limits": lambda _: (228 << 10, 65536, 1024),
            "_l2_size": lambda _: 50 << 20,
            "_scan_workspace_budget": lambda _: 128 << 20,
            "_resident": lambda _, k, smem: min(
                64 // k.warps, (228 << 10) // (1024 + smem), 2 if k.vertical and k.backward else 16
            ),
        }
        for name, fn in limits.items():
            stack.enter_context(patch.object(scans, name, fn))
        stack.enter_context(patch.object(torch.cuda, "get_device_capability", lambda _: (9, 0)))
        for label, h, w, m, c in SHAPES[:2]:
            for op, v, b in OPS:
                src = SimpleNamespace(
                    dtype=torch.bfloat16,
                    is_cuda=True,
                    device=torch.device("cuda"),
                    numel=lambda: 1024 * h * w * m * (4 if v else 1),
                )
                launch = scans._plan_launch(
                    src,
                    (1024, h, w, m),
                    backward=b,
                    vertical=v,
                    coarse=v and c,
                    hidden=8 if v else 0,
                    precise=v and not b,
                    compact=b and c,
                    interleaved=(True, True),
                )
                yield label, op, w, m, launch


def compile_source(src, folder, flags):
    name = hashlib.sha256(src.encode()).hexdigest()[:24]
    path = folder / (name + ".cu")
    path.write_text(src)
    cmd = [
        "nvcc",
        "-std=c++17",
        "-arch=sm_90a",
        *flags,
        "-cubin",
        "-Xptxas=-v",
        str(path),
        "-o",
        str(path) + ".cubin",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    sass = subprocess.check_output(["cuobjdump", "-sass", str(path) + ".cubin"], text=True)
    resource = subprocess.check_output(["cuobjdump", "-res-usage", str(path) + ".cubin"], text=True)

    def res(field):
        return int(re.search(rf"\b{field}:(\d+)", resource)[1])

    return dict(
        regs=res("REG"),
        stack=res("STACK"),
        local=res("LOCAL"),
        static=res("SHARED"),
        LDL=len(re.findall(r"\bLDL\b", sass)),
        STL=len(re.findall(r"\bSTL\b", sass)),
        LDS128=sass.count("LDS.128"),
        SHFL=sass.count("SHFL"),
        LDG=len(re.findall(r"\bLDG\.", sass)),
        STG=len(re.findall(r"\bSTG\.", sass)),
        LDG64=sass.count("LDG.E.64"),
        STG64=sass.count("STG.E.64"),
        LDG32=len(re.findall(r"\bLDG\.E(?:\.STRONG\.SM)?\s", sass)),
        STG32=len(re.findall(r"\bSTG\.E(?:\.STRONG\.SM)?\s", sass)),
    )


def audit():
    rows = []
    with tempfile.TemporaryDirectory() as tmp, concurrent.futures.ThreadPoolExecutor(6) as pool:
        jobs = {}
        for label, op, w, m, launch in offline_plans():
            if op not in ("HF", "VB"):
                continue
            arms = list(VARIANTS)
            reference = os.environ.get("W66_W65_REFERENCE")
            if reference:
                from runpy import run_path
                previous = run_path(reference, run_name="lnet._w65_reference")
                arms.append(previous["ARM"])
            if op == "HF":
                arms.append(fp.HF_STREAMING)
            for variant in arms:
                key = launch.key
                source = (
                    scans._kernel_source(key)
                    if variant == VARIANTS[0]
                    else previous["kernel_source"](key, variant, w, w, m) if variant == "w65_split_bf16" else fp.kernel_source(
                        key,
                        variant,
                        w,
                        w,
                        m,
                        launch.workers == 1024 * (w),
                    )
                )
                if op == "VB" and variant == fp.ARM:
                    assert key.cache_source
                    column = source.split("void column_backward(", 1)[1].split("#undef CACHED_SOURCE_AT", 1)[0]
                    ascending = column.split("adj_a = adj_b = {};", 1)[1]
                    assert "CACHED_SOURCE_AT(j)" in ascending
                    assert "seg_src[I(y-int(COARSE))*THREADS]=injection" not in column
                if variant != "w57_lockin":
                    baseline = scans._kernel_source(key)
                    # Existing source includes dead templates. Compare intrinsic counts
                    # and reject runtime division by either packed scale in the edits.
                    assert source.count("__fdividef(") == baseline.count("__fdividef(")
                    assert not re.search(
                        r"/\s*(?:c\.)?(?:scale|inv_s|ds_scale|row_scale)\b", source
                    )
                    assert source.count("path.input_group(") == baseline.count("path.input_group(")
                    assert source.count("tanh") == baseline.count("tanh")
                if source not in jobs:
                    jobs[source] = pool.submit(
                        compile_source, source, Path(tmp), fp.compile_flags(key, variant, w)
                    )
                rows.append(
                    (
                        dict(
                            shape=label,
                            op=op,
                            variant=variant,
                            threads=32 * key.warps,
                            smem=launch.plan.smem,
                            workers=launch.workers,
                            partials=launch.partials,
                            partial_bytes=launch.partials * m * (94 if op == "VB" else 4) * 4
                            if op in ("VB", "HB")
                            else 0,
                        ),
                        jobs[source],
                    )
                )
        result = []
        for row, future in rows:
            row.update(future.result())
            warps = row["threads"] // 32
            reg_warp = ((row["regs"] * 32 + 255) // 256) * 256
            blocks = min(
                32,
                64 // warps,
                (65536 // 4 // reg_warp) * 4 // warps,
                (228 << 10) // (max(1024, ((row["smem"] + row["static"] + 255) // 256) * 256)),
            )
            row["active_warps_estimate"] = warps * blocks
            result.append(row)
        return result



def test_sm90a_audit():
    rows = audit()
    for row in rows:
        print(row)
        assert not any(row[k] for k in ("stack", "LDL", "STL")), row
        if row["op"] == "VB":
            assert row["LDS128"] > 0, row
    for shape in ("56d", "56c"):
        hf = next(r for r in rows if r["shape"] == shape and r["op"] == "HF" and r["variant"] == fp.ARM)
        assert hf["SHFL"] == 0
        assert hf["STG64"] > 0
        vb = {r["variant"]: r for r in rows if r["shape"] == shape and r["op"] == "VB"}
        assert vb[fp.ARM]["LDG"] <= vb["w57_lockin"]["LDG"], vb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda t:t[0])
def test_hopper(shape, monkeypatch):
    if torch.cuda.get_device_capability()!=(9,0):
        pytest.skip("requires Hopper")
    for k,v in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k,v)
    _,h,w,m,c=shape
    rows=bench.benchmark(1,1,1,[(*shape,1)])["shapes"][shape[0]]["arms"]
    for arm,row in rows.items():
        assert row["pipeline"]["graph_finite"]
        assert row["parity"]["min_cos"]>=0.995
        assert row["parity"]["max_rel_l2"] < 0.1
        if arm==fp.ARM:
            assert row["parity"]["control_close"]
