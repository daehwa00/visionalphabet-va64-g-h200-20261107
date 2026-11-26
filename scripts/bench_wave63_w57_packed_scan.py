"""Compile the wave57 and packed SM90a kernels without opening a CUDA context."""

import argparse
import concurrent.futures
from functools import lru_cache, partial
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from bench_scan_micro import MODEL_SHAPES, _time
from h200.bench_h200 import lockin_arm_env
from lnet import w57_packed_scan as fp
from lnet import general_cuda_scans as scans
from lnet.w57_packed_scan import kernel_source

SHAPES = [
    (f"{h}{suffix}", h, h, 128 if h == 7 else 160, suffix == "c")
    for h in (56, 28, 14, 7)
    for suffix in ("d", "c")
    if h != 7 or suffix == "d"
] + [("odd", 13, 17, 33, False)]
OPS = [("HF", False, False), ("VF", True, False), ("VB", True, True), ("HB", False, True)]
VARIANTS = ("w57_lockin", "packed_bf16", "packed_fp8")


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
        for label, h, w, m, c in SHAPES:
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
        E4M3=sass.count(".E4M3"),
        E5M2=sass.count(".E5M2"),
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
            for variant in VARIANTS:
                key = launch.key
                source = (
                    scans._kernel_source(key)
                    if variant == VARIANTS[0]
                    else kernel_source(
                        key,
                        variant,
                        w,
                        13 if label == "odd" else w,
                        m,
                        launch.workers == 1024 * (13 if label == "odd" else w),
                    )
                )
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


def metrics(ref, val):
    a, b = ref.detach().double().flatten(), val.detach().double().flatten()
    return dict(
        cos=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        rel_l2=float((a - b).norm() / a.norm().clamp_min(1e-30)),
    )


def horizontal(x, a, g):
    out = []
    for reverse in (False, True):
        z = torch.zeros_like(x[:, :, 0])
        row = []
        coeff = a.conj() if reverse else a
        gain = g.conj() if reverse else g
        for t in range(x.shape[2]):
            k = x.shape[2] - 1 - t if reverse else t
            z = coeff * z + gain * x[:, :, k]
            row.append(z)
        out.append(torch.stack(row[::-1] if reverse else row, 2))
    return out


def vertical(states, a, g, aux, path, c, reverse_states=None):
    step = 1 + c
    width = states[0].shape[2] // step
    vals = []
    for rev in (False, True):
        p, n = reverse_states if rev and reverse_states is not None else states
        z = torch.stack([p[:, :, c : width * step : step], n[:, :, : width * step : step]], -1)
        acc = torch.zeros_like(z[:, 0])
        seq = []
        for t in range(z.shape[1]):
            y = z.shape[1] - 1 - t if rev else t
            acc = (a.conj() if rev else a)[:, None] * acc + (g.conj() if rev else g)[:, None] * z[
                :, y
            ]
            if y % step == (0 if rev else step - 1) and y // step < z.shape[1] // step:
                vx = torch.stack(
                    (aux[0][0, c : width * step : step], aux[0][1, : width * step : step]), -1
                )
                seq.append(acc * (vx * aux[1][int(rev), y, :, None]).clamp_min(1e-6).rsqrt())
        vals.append(torch.stack(seq[::-1] if rev else seq, 1))
    q = torch.cat(vals, -1)
    features = torch.cat([q.real, q.imag], -1)
    wi, bi, wo, bo = path
    act = torch.nn.functional.silu(torch.einsum("...mk,mjk->...mj", features, wi) + bi)
    both = torch.einsum("...mj,mkj->...mk", act, wo) + bo
    return torch.cat(both.unbind(-1), -1)


def pack(planes, scale, dtype):
    """CPU/device oracle. Both arguments are in logical BHWM plane order."""
    _, b, h, w, m = planes.shape
    n = (m + 31) // 32
    padded = torch.nn.functional.pad(planes, (0, 32 * n - m)).reshape(4, b, h, w, n, 32)
    raw = padded.permute(1, 4, 3, 2, 5, 0) / scale
    limit = torch.finfo(dtype).max
    codes = raw.clamp(-limit, limit).to(dtype)
    back = codes.float() * scale
    report = {
        "saturation": int((raw.abs() > limit).sum()),
        "underflow": int(((raw != 0) & (back == 0)).sum()),
        "nonfinite": int((~torch.isfinite(back)).sum()),
    }
    words = codes.view(torch.uint8).contiguous().view(torch.uint32).squeeze(-1)
    return words, back.permute(5, 0, 3, 2, 1, 4).reshape(4, b, h, w, n * 32)[..., :m], report


def four(states):
    return torch.stack((states[0].real, states[0].imag, states[1].real, states[1].imag))


def measure(a, b):
    if a.is_complex():
        a, b = torch.view_as_real(a.resolve_conj()), torch.view_as_real(b.resolve_conj())
    return metrics(a, b)


def reference_inputs(h, w, m, device="cpu"):
    torch.manual_seed(6224)

    def bf(x):
        return x.bfloat16().float().requires_grad_()

    real, imag = [bf(torch.randn(1, h, w, m, device=device) * 0.1) for _ in range(2)]
    x = torch.complex(real, imag).detach().requires_grad_()
    a = torch.full((m,), 0.73 + 0.13j, device=device, requires_grad=True)
    g = torch.full((m,), 0.21 + 0.09j, device=device, requires_grad=True)
    aux = [torch.rand(2, s, m, device=device) + 1 for s in (w, h)]
    path = [
        bf(torch.randn(s, device=device) * 0.125) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    return x, a, g, aux, path


def timed(op, warmup, iters, name):
    op()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = op()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op()
    graph.replay()
    values = out if isinstance(out, (list, tuple)) else (out,)
    finite = True
    for t in values:
        if t is None:
            continue
        for chunk in t.reshape(-1).split(1 << 20):
            if chunk.dtype == torch.uint32:
                dtype = torch.float8_e4m3fn if name == "HF" else torch.float8_e5m2
                chunk = chunk.view(torch.uint8).view(dtype).float()
            elif chunk.dtype == torch.uint64:
                chunk = chunk.view(torch.bfloat16)
            if chunk.is_floating_point():
                finite = finite and bool(torch.isfinite(chunk).all())
    if not finite:
        raise RuntimeError("nonfinite graph output")
    return dict(ms=_time(op, warmup, iters), peak_bytes=peak, graph_finite=finite)


def unpack(words, scale, dtype, m):
    if words.dtype == torch.uint64:
        values = words.unsqueeze(-1).view(torch.bfloat16).float()
    else:
        values = words.unsqueeze(-1).view(torch.uint8).view(dtype).float() * scale
    return values.permute(5, 0, 3, 2, 1, 4).flatten(-2)[..., :m]


@lru_cache(maxsize=1)
def accumulating_shapes():
    shapes = set()
    for label, op, w, m, launch in offline_plans():
        if op == "VB" and "column_adjoint_positive<T" not in scans._kernel_source(launch.key):
            shapes.add((w, m, launch.key.coarse))
    return shapes


def precision(h, w, m, c, device="cpu", variant="packed_fp8"):
    x, a, g, aux, path = reference_inputs(h, w, m, device)
    hs = horizontal(x, a, g)
    vs = [z.detach().requires_grad_() for z in hs]
    output = vertical(vs, a, g, aux, path, c)
    dy = (torch.randn_like(output) * 0.05).bfloat16()
    vr = torch.autograd.grad(output, (a, g, *vs, *path), dy.float())
    hb = torch.autograd.grad(hs, (x, a, g), vr[2:4])
    planes = four(hs)
    step = 1 + c
    dw = w // step

    def compact(raw):
        return torch.cat((raw[:2, :, :, c : dw * step : step], raw[2:, :, :, : dw * step : step]))

    refds = compact(four(vr[2:4]))
    if device == "cpu":

        def tiled(planes):
            return torch.nn.functional.pad(planes, (0, (-m) % 32)).unflatten(-1, (-1, 32))

        def roundtrip(v, scale, dtype):
            if variant == "packed_bf16":
                packed = (
                    tiled(v.bfloat16())
                    .permute(1, 4, 3, 2, 5, 0)
                    .contiguous()
                    .view(torch.uint64)
                    .squeeze(-1)
                )
                return unpack(packed, None, None, m), 0
            words, restored, stats = pack(v, scale, dtype)
            assert torch.equal(restored, unpack(words, scale, dtype, m))
            return restored, stats["saturation"]

        pos = tiled(planes[:2]).abs().amax((0, 3, 5)).transpose(1, 2)
        scale = (pos.clamp_min(1.0e-12) / 112).log2().ceil().exp2()[:, :, None, :, None, None]
        # HF stages the positive pair in wave57 BF16 before its FP8 store.
        source_planes = torch.cat((planes[:2].bfloat16().float(), planes[2:]))
        st, sat = roundtrip(source_planes, scale, torch.float8_e4m3fn)
        vsc = [torch.complex(st[k], st[k + 1]).detach().requires_grad_() for k in (0, 2)]
        reverse = [z.detach().requires_grad_() for z in vsc]
        val = vertical(vsc, a, g, aux, path, c, reverse)
        grads = torch.autograd.grad(val, (a, g, *vsc, *reverse, *path), dy.float())
        ma = tiled(torch.stack(dy.split(m, -1))).abs().amax((0, 2, 5)).transpose(1, 2)
        ds_scale = (
            ((ma / 128).clamp_min(1.0e-12) / 112).log2().ceil().exp2()[:, :, :, None, None, None]
        )
        pos_grad = compact(four(grads[2:4]))
        neg_grad = compact(four(grads[4:6]))
        legacy_ds = (pos_grad.bfloat16().float() + neg_grad).bfloat16().float()
        if (w, m, c) in accumulating_shapes():
            stored_pos, extra = roundtrip(pos_grad, ds_scale, torch.float8_e5m2)
            sat += extra
        else:
            stored_pos = pos_grad.bfloat16().float()
        ds, sat2 = roundtrip(stored_pos + neg_grad, ds_scale, torch.float8_e5m2)
        if variant == "packed_bf16":
            assert torch.equal(st, planes.bfloat16().float())
            assert torch.equal(ds, legacy_ds)
        padded = torch.zeros_like(planes)
        padded[:2, :, :, c : dw * step : step] = ds[:2]
        padded[2:, :, :, : dw * step : step] = ds[2:]
        hs = horizontal(x, a, g)
        hb_actual = torch.autograd.grad(
            hs, (x, a, g), [torch.complex(padded[k], padded[k + 1]) for k in (0, 2)]
        )
        actual = [st, val, *grads[:2], ds, *grads[6:], *hb_actual]
    else:
        r, i = x.real.bfloat16(), x.imag.bfloat16()
        poles = [z.contiguous() for z in (a.real, a.imag, g.real, g.imag)]
        weights = [v.bfloat16() for v in path]
        if variant == "w57_lockin":
            st = scans.horizontal_forward(poles, r, i)
            out = scans.vertical_forward_packed(poles, st, aux, weights, c, 1e-6, False)
            v = scans.vertical_backward(poles, st, aux, weights, *dy.split(m, -1), c, 1e-6, c)
            hg = scans.horizontal_backward(poles, [r, i], st, v[4], c)
            ds, params = v[4], v[5:]
        else:
            q, s = fp.horizontal(poles, r, i, variant)
            out = fp.vertical(poles, r, (q, s), aux, weights, c, variant)
            v = fp.vertical(poles, r, (q, s), aux, weights, c, variant, dy.split(m, -1))
            hg = fp.horizontal(poles, r, i, variant, (q, s), v[4:6], c)
            ds = unpack(
                v[4],
                None if v[5] is None else v[5][:, :, :, None, None, None],
                torch.float8_e5m2,
                m,
            )
            st = unpack(
                q, None if s is None else s[:, :, None, :, None, None], torch.float8_e4m3fn, m
            )
            params = v[6:]
        dx = torch.complex(*(t.float() for t in hg[-1].split(m, -1)))
        if variant == "packed_bf16":
            ref_st = scans.horizontal_forward(poles, r, i)
            ref_out = scans.vertical_forward_packed(poles, ref_st, aux, weights, c, 1e-6, False)
            ref_v = scans.vertical_backward(
                poles, ref_st, aux, weights, *dy.split(m, -1), c, 1e-6, c
            )
            ref_h = scans.horizontal_backward(poles, [r, i], ref_st, ref_v[4], c)
            control_error = max(
                metrics(lhs, rhs)["rel_l2"]
                for lhs, rhs in zip(
                    [ref_st, ref_out, *ref_v, *ref_h],
                    [st, out, *v[:4], ds, *params, *hg],
                    strict=True,
                )
            )
        actual = [
            st,
            out,
            torch.complex(*v[:2]),
            torch.complex(*v[2:4]),
            ds,
            *params,
            dx,
            torch.complex(*hg[:2]),
            torch.complex(*hg[2:4]),
        ]
    expected = [planes, output, *vr[:2], refds, *vr[4:], *hb]
    names = "states output v_da v_dg d_states dwi dbi dwo dbo dx h_da h_dg".split()
    result = {k: measure(e, v) for k, e, v in zip(names, expected, actual, strict=True)}
    if device == "cuda" and variant == "packed_bf16":
        result["w57_control_rel_l2"] = control_error
    if device == "cpu":
        result["saturation"] = sat + sat2
        if variant == "packed_bf16":
            result["handoff_rel_l2"] = max(
                metrics(planes.bfloat16().float(), st)["rel_l2"],
                metrics(legacy_ds, ds)["rel_l2"],
            )
    return result


def benchmark(batch, warmup, iters):
    os.environ.update(lockin_arm_env("w57_lockin"))
    shapes, totals = {}, {}
    for label, h, w, m, c, count in MODEL_SHAPES:
        pole = [torch.full((m,), v, device="cuda") for v in (0.73, 0.13, 0.21, 0.09)]
        aux = [1 + torch.rand(2, n, m, device="cuda") for n in (w, h)]
        path = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(0.125)
            for shape in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
        ]
        gr, gi = (
            torch.randn(
                batch, h // (1 + c), w // (1 + c), 2 * m, device="cuda", dtype=torch.bfloat16
            )
            .mul_(0.05)
            .split(m, -1)
        )
        x = torch.randn(batch, h, w, 2 * m, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        r, i = x.split(m, -1)
        rows = {}
        for variant in VARIANTS:
            if variant == "w57_lockin":
                hf = lambda: scans.horizontal_forward(pole, r, i)
                vf = lambda s: scans.vertical_forward_packed(pole, s, aux, path, c, 1e-6, False)
                vb = lambda s: scans.vertical_backward(pole, s, aux, path, gr, gi, c, 1e-6, c)
                hb = lambda s, v: scans.horizontal_backward(pole, [r, i], s, v[4], c)
            else:
                hf = lambda: fp.horizontal(pole, r, i, variant)
                vf = lambda s: fp.vertical(pole, r, s, aux, path, c, variant)
                vb = lambda s: fp.vertical(pole, r, s, aux, path, c, variant, (gr, gi))
                hb = lambda s, v: fp.horizontal(pole, r, i, variant, s, v[4:6], c)

            def pipeline():
                s = hf()
                out = vf(s)
                v = vb(s)
                return [out, *v[:4], *v[(5 if variant == "w57_lockin" else 6) :], *hb(s, v)]

            s = hf()
            v = vb(s)
            rows[variant] = {
                op: timed(fn, warmup, iters, op)
                for op, fn in [
                    ("HF", hf),
                    ("VF", partial(vf, s)),
                    ("VB", partial(vb, s)),
                    ("HB", partial(hb, s, v)),
                ]
            }
            del s, v
            rows[variant]["pipeline"] = timed(pipeline, warmup, iters, "pipeline")
            for op, row in rows[variant].items():
                base = rows["w57_lockin"][op]
                row["delta_ms"] = row["ms"] - base["ms"]
                row["delta_peak_MiB"] = (row["peak_bytes"] - base["peak_bytes"]) / 2**20
                totals.setdefault(variant, {}).setdefault(op, 0.0)
                totals[variant][op] += count * row["ms"]
            rows[variant]["parity"] = precision(h, w, m, c, "cuda", variant)
        shapes[label] = dict(calls=count, arms=rows)
    deltas = {
        v: {op: ms - totals["w57_lockin"][op] for op, ms in values.items()}
        for v, values in totals.items()
    }
    return dict(shapes=shapes, weighted_ms=totals, weighted_delta_ms=deltas)


def rounded(v):
    if isinstance(v, float):
        return float(format(v, ".5g"))
    if isinstance(v, dict):
        return {k: rounded(x) for k, x in v.items()}
    if isinstance(v, list):
        return list(map(rounded, v))
    return v


def validate_resources(rows):
    for base, bf, fp8 in zip(rows[::3], rows[1::3], rows[2::3], strict=True):
        for row in (base, bf, fp8):
            assert all(row[k] == 0 for k in ("stack", "local", "LDL", "STL")), row
            assert row["active_warps_estimate"] >= base["active_warps_estimate"], row
            assert all(row[k] == base[k] for k in ("threads", "workers", "partials", "smem")), row
        for row, bits in ((bf, "64"), (fp8, "32")):
            if row["op"] != "HF":
                assert row["LDG" + bits] > 0, row
            if row["op"] in ("VB", "HF"):
                assert row["STG" + bits] > 0, row
            if bits == "32" and row["op"] in ("HF", "VB"):
                assert row["E4M3" if row["op"] == "HF" else "E5M2"] > 0, row
            if row["op"] == "VB" and base["LDS128"]:
                assert row["LDS128"] > 0, row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-audit", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()
    if args.offline_audit:
        torch.set_num_threads(2)
        rows = audit()
        checks = {
            label: {v: precision(h, w, m, c, variant=v) for v in VARIANTS[1:]}
            for label, h, w, m, c in SHAPES
        }
        result = dict(resources=rows, cpu_model=checks)
    else:
        result = benchmark(args.batch, args.warmup, args.iters)
    payload = json.dumps(rounded(result), separators=(",", ":"), allow_nan=False)
    if args.json:
        args.json.write_text(payload + "\n")
    else:
        print(payload)
    if args.offline_audit:
        validate_resources(rows)
        assert all(m["packed_bf16"]["handoff_rel_l2"] == 0 for m in checks.values())
        assert all(
            x["cos"] >= 0.995
            for modes in checks.values()
            for x in modes["packed_fp8"].values()
            if isinstance(x, dict)
        )
