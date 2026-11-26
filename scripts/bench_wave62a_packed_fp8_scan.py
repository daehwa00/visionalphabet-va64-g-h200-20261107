"""Offline SM90a cubin audit and optional device screening of packed scan handoffs."""

import argparse
from contextlib import ExitStack
from functools import cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from bench_scan_micro import MODEL_SHAPES, make_inputs, _time
from h200.bench_h200 import lockin_arm_env
from lnet import general_cuda_scans as scans
from lnet import packed_fp8_scan as fp

EDGE = [("odd", 13, 17, 33, False, 1), ("odd_c", 13, 17, 33, True, 1)]


def resources(image, name, threads, dynamic=0):
    with tempfile.TemporaryDirectory() as path:
        cubin = Path(path) / "scan.cubin"
        cubin.write_bytes(image)
        resources = subprocess.check_output(["cuobjdump", "-res-usage", str(cubin)], text=True)
        sass = subprocess.check_output(["cuobjdump", "-sass", "-fun", name, str(cubin)], text=True)
    line = re.search(r"Function " + name + r":\s*([^\n]+)", resources)[1]
    fields = {k: int(v) for k, v in re.findall(r"(\w+):(\d+)", line)}
    reg = fields["REG"]
    shared = fields["SHARED"] + dynamic
    warps = threads // 32
    blocks = min(
        32,
        64 // warps,
        65536 // (((reg + 7) // 8) * 256 * warps),
        (228 * 1024) // max(1024, ((shared + 255) // 256) * 256),
    )
    assert "Function : " + name in sass
    ldg32 = len(re.findall(r"\bLDG\.E\s", sass))
    stg32 = len(re.findall(r"\bSTG\.E\s", sass))
    if name in (fp.VF, fp.PURE, fp.HB, fp.SEG, fp.FULL):
        assert ldg32 > 0, name
    if name in (fp.HF, fp.SEG, fp.FULL):
        assert stg32 > 0, name
    if name.startswith("w62_"):
        assert "HMMA" not in sass and "HGMMA" not in sass
    result = dict(
        ldg32=ldg32,
        stg32=stg32,
        regs=reg,
        smem=shared,
        threads=threads,
        aw_sm=blocks * warps,
        stack=fields["STACK"],
        local=fields["LOCAL"],
        spill_loads=len(re.findall(r"\bLDL\b", sass)),
        spill_stores=len(re.findall(r"\bSTL\b", sass)),
    )
    return result


@cache
def baseline_image(key):
    ptx = scans._compile_ptx(key, (9, 0)).value
    with tempfile.TemporaryDirectory() as path:
        src = Path(path) / "baseline.ptx"
        src.write_bytes(ptx)
        cubin = src.with_suffix(".cubin")
        subprocess.run(["ptxas", "-arch=sm_90a", str(src), "-o", str(cubin)], check=True)
        return cubin.read_bytes()


def baseline(h, w, m, c):
    # Match the pinned offline planner contract in test_scan_wave57a.py.
    with ExitStack() as ctx:
        ctx.enter_context(patch.dict(os.environ, lockin_arm_env("w57_lockin")))
        for fn, value in [
            ("device_limits", (132, 227 << 10, 64)),
            ("_sgs_device_limits", (228 << 10, 65536, 1024)),
            ("_l2_size", 50 << 20),
            ("_scan_workspace_budget", 128 << 20),
        ]:
            ctx.enter_context(patch.object(scans, fn, return_value=value))
        ctx.enter_context(
            patch.object(
                scans,
                "_resident",
                side_effect=lambda _, k, s: min(
                    64 // k.warps, (228 << 10) // (1024 + s), 2 if k.vertical and k.backward else 16
                ),
            )
        )
        ctx.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)))
        result = {}
        for name, v, bw in [
            ("vf", True, False),
            ("vb", True, True),
            ("hf", False, False),
            ("hb", False, True),
        ]:
            src = SimpleNamespace(
                dtype=torch.bfloat16,
                is_cuda=True,
                device=torch.device("cuda"),
                numel=lambda: 1024 * h * w * m * (4 if v else 1),
            )
            launch = scans._plan_launch(
                src,
                (1024, h, w, m),
                vertical=v,
                backward=bw,
                hidden=8 if v else 0,
                precise=v and not bw,
                coarse=v and c,
                compact=bw and c,
                interleaved=(True, True),
            )
            key = launch.key
            result["w57_" + name] = resources(
                baseline_image(key), scans._kernel_name(key), 32 * key.warps, launch.plan.smem
            )
            result["w57_" + name].update(
                cache_line=key.cache_line,
                vec2=key.horizontal_vec2,
                half_replay=key.horizontal_half_replay,
            )
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


def vertical(states, a, g, aux, path, c):
    step = 1 + c
    p, n = states
    width = p.shape[2] // step
    z = torch.stack([p[:, :, c : width * step : step], n[:, :, : width * step : step]], -1)
    vals = []
    for rev in (False, True):
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


def quantcomplex(z, dtype):
    ri = torch.view_as_real(z.resolve_conj())
    return torch.view_as_complex(ri.to(dtype).float().contiguous())


@torch.no_grad()
def column_mirror(src, a, g, aux, path, dy, step, full):
    """Execute the vertical adjoint with the same segment boundaries and storage casts as CUDA."""
    b, h, ow, m, _ = src.shape
    span = h if full else (7 if h <= 7 else 14)
    blocks = list(range(0, h, span))
    oh = h // step
    gr, gi = dy.split(m, -1)

    def maxgrad(v):
        return torch.nn.functional.pad(v, (0, (-m) % 32)).unflatten(-1, (-1, 32)).abs().amax((1, 4))

    scales = 2 ** torch.ceil(
        torch.log2(torch.maximum(maxgrad(gr), maxgrad(gi)).clamp_min(1e-12) / 64)
    )
    scale = scales.repeat_interleave(32, -1)[..., :m, None]
    counts = {
        k: dict(saturation=0, underflow=0, nonfinite=0) for k in ("d_states_positive", "d_states")
    }

    def cast(z, s=1, key=None):
        raw = torch.view_as_real((z / s).resolve_conj())
        q = raw.clamp(-57344, 57344).to(torch.float8_e5m2).float()
        if key:
            counts[key]["saturation"] += int((raw.abs() > 57344).sum())
            counts[key]["underflow"] += int(((q == 0) & (raw != 0)).sum())
            counts[key]["nonfinite"] += int((~torch.isfinite(q)).sum())
        return torch.view_as_complex(q.contiguous()) * s

    def bf(z):
        return quantcomplex(z, torch.bfloat16)

    a, g = a[:, None], g[:, None]
    z = torch.zeros_like(src[:, 0])
    vx = torch.stack((aux[0][0, step - 1 : ow * step : step], aux[0][1, : ow * step : step]), -1)

    def gain(y, rev):
        return (vx * aux[1][rev, y, :, None]).clamp_min(1e-6).rsqrt()

    weights = [torch.zeros((b, ow, *v.shape), dtype=torch.float16) for v in path]
    wi, bi, wo, bo = path

    def project(pos, neg, y, add):
        gp, gn = gain(y + step - 1, 0), gain(y, 1)
        v = torch.cat((bf(bf(pos) * gp), bf(bf(neg) * gn)), -1)
        f = torch.cat((v.real, v.imag), -1)
        pre = torch.einsum("...mk,mjk->...mj", f, wi) + bi
        s = pre.sigmoid()
        grad = torch.stack((gr[:, y // step], gi[:, y // step]), -1)
        dz = torch.einsum("...mj,mjk->...mk", grad, wo) * s * (1 + pre * (1 - s))
        dp = torch.einsum("...mj,mjk->...mk", dz, wi)
        v = torch.complex(dp[..., :4], dp[..., 4:])
        if add:
            vals = [
                dz.unsqueeze(-1) * f.unsqueeze(-2),
                dz,
                grad.unsqueeze(-1) * (pre * s).unsqueeze(-2),
                grad,
            ]
            for dst, val in zip(weights, vals, strict=True):
                div = (scale * 64).unsqueeze(-1) if val.ndim == 5 else scale * 64
                dst.add_((val / div).half())
        return bf(bf(v[..., :2]) * gp), bf(bf(v[..., 2:]) * gn)

    pc, nc, ac = {0: z}, {h: z}, {h: z}
    pos = {}
    f = z
    for y in range(h):
        f = a * f + g * src[:, y]
        if full:
            pos[y] = cast(f)
        elif (y + 1) % span == 0:
            pc[y + 1] = cast(f)

    def restore(lo, hi):
        f = pc[lo]
        for y in range(lo, hi):
            f = a * f + g * src[:, y]
            pos[y] = cast(f)

    neg, ta, tg, pa = z, z, z, z
    da, dg = torch.zeros_like(z), torch.zeros_like(z)
    ipos, ineg = {}, {}
    for lo in reversed(blocks):
        hi = min(h, lo + span)
        if not full:
            restore(lo, hi)
            nc[hi] = cast(neg)
            ac[hi] = cast(pa, scale)
        ip = z
        for y in range(hi - 1, lo - 1, -1):
            ta, tg = a.conj() * ta + neg, a.conj() * tg + src[:, y]
            neg = a.conj() * neg + g.conj() * src[:, y]
            if y % step:
                continue
            valid = y // step < oh
            ip = z
            if valid:
                ip, inn = project(pos[y + step - 1], neg, y, True)
                da += (inn * ta.conj()).conj()
                dg += (inn * tg.conj()).conj()
                if full:
                    ineg[y] = cast(inn, scale)
            for r in reversed(range(step if valid else 1)):
                j = y + r
                pa = a.conj() * pa + (ip if r == step - 1 else z)
                prev = pos[j - 1] if j > lo else pc[lo]
                da += pa * prev.conj()
                dg += pa * src[:, j].conj()
                if full:
                    ipos[j] = cast(g.conj() * pa, scale, "d_states_positive")
    adj = z
    dout = []
    for lo in blocks:
        hi = min(h, lo + span)
        if not full:
            restore(lo, hi)
            neg, pa = nc[hi], ac[hi]
            for y in range(hi - 1, lo - 1, -1):
                neg = a.conj() * neg + g.conj() * src[:, y]
                if y % step:
                    continue
                valid = y // step < oh
                ip = z
                if valid:
                    ip, inn = project(pos[y + step - 1], neg, y, False)
                    ineg[y] = cast(inn, scale)
                for r in reversed(range(step if valid else 1)):
                    pa = a.conj() * pa + (ip if r == step - 1 else z)
                    ipos[y + r] = cast(g.conj() * pa, scale, "d_states_positive")
        for y in range(lo, hi):
            adj = a * adj + ineg.get(y, z)
            dout.append(cast(ipos[y] + g * adj, scale, "d_states"))
    ds = torch.stack(dout, 1)
    dw = [
        (v.float() * ((scale * 64).unsqueeze(-1) if v.ndim == 5 else scale * 64)).sum((0, 1))
        for v in weights
    ]
    return da.sum((0, 1, 3)), dg.sum((0, 1, 3)), four(ds.unbind(-1)), dw, counts


def simulate(h, w, m, c, full=False):
    x, a, g, aux, path = reference_inputs(h, w, m)
    step = 1 + c
    st = horizontal(x, a, g)
    plane = four(st).detach()
    buffered = plane.bfloat16().float()
    padded = torch.nn.functional.pad(plane, (0, (-m) % 32)).unflatten(-1, (-1, 32))
    rs = 2 ** torch.ceil(torch.log2(padded.abs().amax((0, 3, 5)).clamp_min(1e-12) / 448))
    _, saved, state_counts = pack(
        buffered, rs.permute(0, 2, 1)[:, :, None, :, None, None], torch.float8_e4m3fn
    )
    qs = [torch.complex(saved[k], saved[k + 1]) for k in (0, 2)]
    refout = vertical(st, a, g, aux, path, c)
    dy = torch.randn_like(refout).bfloat16().float() * 0.0625
    # Detach horizontal recurrences to isolate the vertical pole gradients.
    vs = [z.detach().requires_grad_() for z in st]
    vr = torch.autograd.grad(vertical(vs, a, g, aux, path, c), (a, g, *vs, *path), dy)
    rawgrad = four(vr[2:4])

    def compact(z):
        n = w // step
        return torch.cat((z[:2, :, :, c : n * step : step], z[2:, :, :, 0 : n * step : step]))

    refds = compact(rawgrad)
    hgrad = torch.autograd.grad(st, (x, a, g), vr[2:4])
    qst = [z.detach().requires_grad_() for z in qs]
    ow = w // step
    src = torch.stack((qst[0][:, :, c : ow * step : step], qst[1][:, :, : ow * step : step]), -1)
    vx = torch.stack((aux[0][0, c : ow * step : step], aux[0][1, : ow * step : step]), -1)

    def sweep(reverse):
        coeff = (a.conj() if reverse else a)[:, None]
        gain = (g.conj() if reverse else g)[:, None]
        z = torch.zeros_like(src[:, 0])
        seq = []
        for i in range(h):
            y = h - 1 - i if reverse else i
            z = coeff * z + gain * src[:, y]
            if y % step == (0 if reverse else step - 1) and y // step < h // step:
                v = z
                v = v * (vx * aux[1][int(reverse), y, :, None]).clamp_min(1e-6).rsqrt()
                if not reverse:
                    v = quantcomplex(v, torch.float8_e4m3fn)
                seq.append(v)
        return torch.stack(seq[::-1] if reverse else seq, 1)

    def project(p, n):
        z = torch.cat((p, n), -1)
        features = torch.cat((z.real, z.imag), -1)
        wi, bi, wo, bo = path
        hval = torch.nn.functional.silu(torch.einsum("...mk,mjk->...mj", features, wi) + bi)
        return torch.cat((torch.einsum("...mj,mkj->...mk", hval, wo) + bo).unbind(-1), -1)

    pred = project(sweep(False), sweep(True)).bfloat16().float()
    va, vg, dout, weights, counts = column_mirror(src, a, g, aux, path, dy, step, full)
    up = torch.zeros_like(rawgrad)
    up[:2, :, :, c : ow * step : step] = dout[:2]
    up[2:, :, :, : ow * step : step] = dout[2:]
    left = torch.zeros_like(x[:, :, 0])
    right = torch.zeros_like(left)
    zr, zi = torch.complex(up[0], up[1]), torch.complex(up[2], up[3])
    rights = []
    for k in range(w):
        right = a * right + zi[:, :, k]
        rights.append(quantcomplex(right, torch.bfloat16))
    dx = []
    dpa = torch.zeros_like(left)
    dpg = torch.zeros_like(left)
    for k in reversed(range(w)):
        left = a.conj() * left + zr[:, :, k]
        right = rights[k]
        if k:
            dpa += left * qs[0][:, :, k - 1].conj()
        if k + 1 < w:
            dpa += (right * qs[1][:, :, k + 1].conj()).conj()
        dpg += left * x[:, :, k].conj() + (right * x[:, :, k].conj()).conj()
        dx.append(quantcomplex(g.conj() * left + g * right, torch.bfloat16))
    hb = (torch.stack(dx[::-1], 2), dpa.sum((0, 1)), dpg.sum((0, 1)))
    expected = [plane, refout, *vr[:2], refds, *vr[4:], *hgrad]
    actual = [saved, pred, va, vg, dout, *weights, *hb]
    keys = "states output v_da v_dg d_states dwi dbi dwo dbo dx h_da h_dg".split()
    precision = {k: measure(r, v) for k, r, v in zip(keys, expected, actual, strict=True)}
    return precision, dict(states=state_counts, **counts)


def offline():
    torch.set_num_threads(1)
    result = {}
    for label, h, w, m, c, count in (*MODEL_SHAPES, *EDGE):
        rows = baseline(h, w, m, c)
        image = fp.image(h, w, c)
        for name in fp.NAMES:
            row = resources(image, name, 128, fp.shared_bytes(name, h, w, c))
            assert all(
                row[key] == 0 for key in ("stack", "local", "spill_loads", "spill_stores")
            ), (label, name, row)
            if label == "56d" and name in (fp.HF, fp.HB, fp.SEG, fp.VF, fp.BF16):
                assert row["aw_sm"] >= 16, (label, name, row)
            rows[name] = row
        precision, counts = simulate(h, w, m, c)
        full_precision, full_counts = simulate(h, w, m, c, full=True)
        assert all(v["cos"] >= 0.995 for v in full_precision.values()), (label, full_precision)
        assert all(v["nonfinite"] == 0 for v in full_counts.values())
        assert all(v["cos"] >= 0.995 for v in precision.values()), (label, precision)
        assert all(v["nonfinite"] == 0 for v in counts.values())
        for arm in (precision, full_precision):
            assert arm["states"]["cos"] >= 0.999 and arm["d_states"]["cos"] >= 0.997
        result[label] = dict(
            resources=rows,
            precision=precision,
            counts=counts,
            full_precision=full_precision,
            full_counts=full_counts,
            calls=count,
        )
        print(label, rows, precision, counts, flush=True)
    return result


def device_precision(h, w, m, c):
    x, a, g, aux, path = reference_inputs(h, w, m, "cuda")
    r, i = x.real.bfloat16(), x.imag.bfloat16()
    poles = [v.contiguous() for v in (a.real, a.imag, g.real, g.imag)]
    weights = [v.bfloat16() for v in path]
    st = horizontal(x, a, g)
    vs = [z.detach().requires_grad_() for z in st]
    output = vertical(vs, a, g, aux, path, c)
    dy = (torch.randn_like(output) * 0.05).bfloat16()
    gr, gi = dy.split(m, -1)
    vr = torch.autograd.grad(output, (a, g, *vs, *path), dy.float())
    hb = torch.autograd.grad(st, (x, a, g), vr[2:4])

    def unpack(q, s, dtype):
        v = q.unsqueeze(-1).view(torch.uint8).view(dtype).float() * s
        return v.permute(5, 0, 3, 2, 1, 4).flatten(-2)[..., :m]

    q, s = fp.w62_hscan_fwd_packed(poles, r, i)
    states = unpack(q, s[:, :, None, :, None, None], torch.float8_e4m3fn)
    out = fp.w62_vscan_fwd_packed(poles, q, s, aux, weights, c)
    step = 1 + c
    n = w // step
    raw = four(vr[2:4])
    refds = torch.cat((raw[:2, :, :, c : n * step : step], raw[2:, :, :, : n * step : step]))
    report = {}
    for mode in (fp.SEG, fp.FULL, fp.BF16):
        src = four(st).bfloat16().contiguous() if mode == fp.BF16 else q
        v = fp.w62_vscan_bwd_packed(poles, src, s, aux, weights, gr, gi, c, mode=mode)
        vc = lambda k: torch.complex(v[k], v[k + 1])
        actual = [
            states,
            out,
            vc(0),
            vc(2),
            v[4]
            if mode == fp.BF16
            else unpack(v[4], v[5][:, :, :, None, None, None], torch.float8_e5m2),
            *v[6:],
        ]
        expected = [four(st), output, *vr[:2], refds, *vr[4:]]
        names = "states output v_da v_dg d_states dwi dbi dwo dbo".split()
        if mode != fp.BF16:
            hg = fp.w62_hscan_bwd_packed(poles, r, i, q, s, v[4], v[5])
            dx = torch.complex(*(z.float() for z in hg[-1].split(m, -1)))
            actual.extend((dx, torch.complex(*hg[:2]), torch.complex(*hg[2:4])))
            expected.extend(hb)
            names.extend(("dx", "h_da", "h_dg"))
        report[mode] = {k: measure(a, b) for k, a, b in zip(names, expected, actual, strict=True)}
    return report


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
    finite = all(
        bool(torch.isfinite(t).all()) for t in values if t is not None and t.is_floating_point()
    )
    for t in values:
        if t is not None and t.dtype == torch.uint32:
            dtype = torch.float8_e4m3fn if name == fp.HF else torch.float8_e5m2
            finite = finite and bool(torch.isfinite(t.view(torch.uint8).view(dtype).float()).all())
    del graph, out, values
    if not finite:
        raise RuntimeError("nonfinite graph output")
    return dict(ms=_time(op, warmup, iters), peak_bytes=peak, graph_finite=finite)


def benchmark(b, warmup, iters):
    os.environ.update(lockin_arm_env("w57_lockin"))
    totals, shapes = {}, {}
    for label, h, w, m, c, count in MODEL_SHAPES:
        pole, unused, aux, path, gr, gi = make_inputs(b, h, w, m, c, torch.bfloat16)
        del unused
        x = torch.randn(b, h, w, 2 * m, device="cuda", dtype=torch.bfloat16) * 0.1
        r, i = x.split(m, -1)
        hf = lambda: scans.horizontal_forward(pole, r, i)
        pf = lambda: fp.w62_hscan_fwd_packed(pole, r, i)
        vf = lambda st: scans.vertical_forward_packed(pole, st, aux, path, c, 1e-6, planar=False)
        pv = lambda state, pure=False: fp.w62_vscan_fwd_packed(
            pole, *state, aux, path, c, pure=pure
        )
        vb = lambda st: scans.vertical_backward(pole, st, aux, path, gr, gi, c, 1e-6, c)
        pb = lambda state, mode: fp.w62_vscan_bwd_packed(
            pole, *state, aux, path, gr, gi, c, mode=mode
        )
        hb = lambda st, d: scans.horizontal_backward(pole, [r, i], st, d, compact=c)
        ph = lambda state, d: fp.w62_hscan_bwd_packed(pole, r, i, *state, *d)
        st, q = hf(), pf()
        gd = vb(st)[4]
        dq = pb(q, fp.SEG)[4:6]

        def pipeline(mode):
            if mode == "w57":
                s = hf()
                out = vf(s)
                adj = vb(s)
                return [out, *adj[:4], *adj[5:], *hb(s, adj[4])]
            s = pf()
            out = pv(s)
            adj = pb(s, mode)
            return [out, *adj[:4], *adj[6:], *ph(s, adj[4:6])]

        ops = {
            "w57_hf": hf,
            fp.HF: pf,
            "w57_vf": lambda: vf(st),
            fp.VF: lambda: pv(q),
            "w57_vb": lambda: vb(st),
            fp.SEG: lambda: pb(q, fp.SEG),
            fp.FULL: lambda: pb(q, fp.FULL),
            fp.BF16: lambda: pb((st, q[1]), fp.BF16),
            "w57_hb": lambda: hb(st, gd),
            fp.HB: lambda: ph(q, dq),
            "w57_pipeline": lambda: pipeline("w57"),
            "packed_seg_pipeline": lambda: pipeline(fp.SEG),
            "packed_full_pipeline": lambda: pipeline(fp.FULL),
        }
        if label in ("56c", "56d"):
            ops[fp.PURE] = lambda: pv(q, True)
        rows = {key: timed(op, warmup, iters, key) for key, op in ops.items()}
        for name, row in rows.items():
            totals[name] = totals.get(name, 0) + count * row["ms"]
        if fp.PURE in rows:
            # Useful global payload bytes. Coarse visits two packed input columns per output column.
            payload = 4 * q[0].numel() + 2 * 8 * b * (h // (1 + c)) * (w // (1 + c)) * m
            rows[fp.PURE]["payload_GB_s"] = payload / rows[fp.PURE]["ms"] / 1e6
        for k in ("packed_seg_pipeline", "packed_full_pipeline"):
            rows[k]["saved_ms"] = rows["w57_pipeline"]["ms"] - rows[k]["ms"]
            rows[k]["saved_peak_bytes"] = rows["w57_pipeline"]["peak_bytes"] - rows[k]["peak_bytes"]
        shapes[label] = {
            "calls": count,
            "timings": rows,
            "precision_fp32": device_precision(h, w, m, c),
        }
    return dict(
        shapes=shapes,
        weighted_ms=totals,
        saved_ms={
            k: totals["w57_pipeline"] - totals[k]
            for k in ("packed_seg_pipeline", "packed_full_pipeline")
        },
    )


def rounded(v):
    if isinstance(v, float):
        return float(format(v, ".5g"))
    if isinstance(v, dict):
        return {k: rounded(x) for k, x in v.items()}
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-audit", action="store_true")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = offline() if args.offline_audit else benchmark(args.batch, args.warmup, args.iters)
    data = json.dumps(rounded(result), separators=(",", ":"), allow_nan=False)
    if args.json:
        args.json.write_text(data + "\n")
    elif not args.offline_audit:
        print(data)
