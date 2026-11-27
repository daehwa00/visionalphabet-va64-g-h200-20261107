# ruff: noqa: ANN201, B023, C408, FBT003, E501, PLC0415, PLW1510, RUF059, S607, T201
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


def offline_plans(shapes=SHAPES[:2], *, plane_outputs=False):
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
        for label, h, w, m, c in shapes:
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
                    interleaved=(not (plane_outputs and v and not b and h != 7),) * 2,
                )
                yield label, op, w, m, launch


def compile_source(src, folder, flags, symbol=None):
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
    select = ["-fun", symbol] if symbol else []
    sass = subprocess.check_output(["cuobjdump", *select, "-sass", str(path) + ".cubin"], text=True)
    resource = subprocess.check_output(["cuobjdump", "-res-usage", str(path) + ".cubin"], text=True)

    if symbol:
        resource = resource.split(f"Function {symbol}:", 1)[1].splitlines()[1]

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


def test_lockin_arms():
    base = bench.lockin_arm_env("w57_lockin")
    assert "VA_W66_SPLIT_BF16" not in base
    for name in ("w66_lockin", "w66_lockin_rev"):
        assert bench.lockin_arm_env(name) == {**base, "VA_W66_SPLIT_BF16": "1"}


def test_dispatch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    path = [SimpleNamespace(shape=(160, 8, 8))] * 4
    for flag in ("0", "1"):
        monkeypatch.setenv("VA_W66_SPLIT_BF16", flag)
        for dtype in (torch.float32, torch.bfloat16):
            for _, h, w, m, coarse in [*SHAPES, ("odd", 55, 56, 160, False)]:
                source = SimpleNamespace(shape=(1,h,w,m), device="cuda", dtype=dtype, is_cuda=True, ndim=4)
                assert fp.enabled(source, path, "coarse_path" if coarse else "full16", 0, swiglu=False) == (
                    flag == "1" and dtype == torch.bfloat16 and h in (28, 56)
                )


def test_profile_names():
    from h200.bench_h200 import profile_bucket
    for _, vertical, backward in OPS:
        key = SimpleNamespace(vertical=vertical, backward=backward)
        assert profile_bucket(fp.profile_name(key)) == (
            ("vscan" if vertical else "hscan") + ("_bwd" if backward else "_fwd")
        )


@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("planar", [False, True])
def test_registered_meta(coarse, planar):
    r, i = torch.empty((2,28,28,320), device="meta", dtype=torch.bfloat16).chunk(2, -1)
    p = [r.new_empty((160,), dtype=torch.float32) for _ in range(4)]
    path = [r.new_empty(s) for s in ((160,8,8), (160,8), (160,2,8), (160,2))]
    states = fp.hforward(p, r, i)
    assert states.shape == (2,5,2,28,28,32)
    assert states.dtype == torch.uint32
    assert states.is_contiguous()
    out = fp.vforward(p, r, states, [], path, coarse, planar, 1e-8)
    k = 28 // (1 + coarse)
    assert out.shape == ((2,2,k,k,160) if planar else (2,k,k,320))
    assert out.dtype == r.dtype
    assert out.is_contiguous()
    for plane in fp.vforward_planes(p, r, states, [], path, coarse, 1e-8):
        assert plane.shape == (2,k,k,160)
        assert plane.is_contiguous()
        assert plane.dtype == r.dtype
    gr, gi = out.unbind(0) if planar else out.chunk(2, -1)
    grads = fp.scan_backward(p, p, r, i, states, [], path, gr, gi, coarse, 1e-8)
    for g, expected in zip(grads, [*p, *p, r.new_empty((2,28,28,320)), *path], strict=True):
        assert (g.shape, g.stride(), g.dtype) == (expected.shape, expected.stride(), expected.dtype)


@pytest.mark.parametrize(("separate", "recompute"), [(False, True), (True, False)])
def test_production_fake_pipeline(monkeypatch, separate, recompute):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from lnet import product_scan_pipeline as pipeline

    # Exercise the production composition using CPU fake tensors and real op registrations.
    monkeypatch.setattr(fp, "enabled", lambda *_: True)
    with FakeTensorMode():
        drive = torch.empty((2, 28, 28, 320), dtype=torch.bfloat16)
        inputs = [drive.new_empty((1,1,1,160), dtype=torch.float32) for _ in range(8)] + [drive]
        path = [drive.new_empty(s) for s in ((160,8,8), (160,8), (160,2,8), (160,2))]
        primary, secondary, desc, saved = pipeline._run_packed_cuda_forward(
            inputs, path, "coarse_path", 1e-8, 0, recompute, False,
            packed_output=not separate, planar_output=False, squeeze_collapsed=separate,
        )
        gr, gi = (primary, secondary) if separate else primary.chunk(2, -1)
        grads = pipeline._run_packed_cuda_backward(
            inputs, path, saved, desc, gr, gi, desc, "coarse_path", 1e-8, 0, False,
            packed_output=not separate, planar_output=False,
        )
        assert [s.dtype for s in saved] == ([] if recompute else [torch.uint32])
        for grad, source in zip(grads, [*inputs, *path], strict=True):
            assert (grad.shape, grad.stride(), grad.dtype) == (source.shape, source.stride(), source.dtype)


def test_ablation17_scan_graphs(monkeypatch):
    from test_w66_split_bf16_scan import model_scan_graph

    ops = model_scan_graph(monkeypatch, fp8=False, bf16=False)
    assert len(ops) == 11
    assert all(name.endswith(("planes_v1.default", "storage_v2.default")) for name, _, _ in ops)


def test_storage_graph_target(monkeypatch):
    from lnet.product_scan_pipeline import run_product_scan_path_collapse_pipeline_packed

    drive = torch.zeros((1, 2, 2, 4))
    pole = (torch.ones((1, 1, 1, 2)),) * 4
    path = tuple(torch.zeros(s) for s in ((2, 8, 8), (2, 8), (2, 2, 8), (2, 2)))
    for flag in ("0", "1"):
        monkeypatch.setenv("VA_W66_SPLIT_BF16", flag)
        result = torch._dynamo.explain(run_product_scan_path_collapse_pipeline_packed)(
            pole,
            pole,
            drive,
            path,
            gain_normalization="pointwise",
        )
        name = "product_scan_pipeline_storage_v2" + ("_w66" if flag == "1" else "")
        assert result.graph_count == 1
        assert getattr(torch.ops.lnet, name).default in result.ops_per_graph[0]
