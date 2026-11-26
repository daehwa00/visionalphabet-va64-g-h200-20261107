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
from lnet import scan_fp8_handoff as fp

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
    result = dict(
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
        for name, v, bw in [("vf", True, False), ("hf", False, False), ("hb", False, True)]:
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
                precise=v,
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


def traffic(b, h, w, m, c):
    n = 4 * b * h * w * m
    grad = 4 * b * h * (w // (1 + c)) * m
    scales = 4 * b * h * ((m + 31) // 32)
    used = 4 * b * h * (w // (1 + c)) * m
    # Logical requested bytes; repeated scale fetches may hit in cache.
    scale_loads = scales * (w // (1 + c))
    return dict(
        state_bf16=n * 2,
        state_fp8=n,
        state_scale=scales,
        hf_write_bf16=2 * n,
        hf_write_fp8=n + scales,
        vf_read_2pass_bf16=4 * used,
        vf_read_a=2 * used,
        vf_read_b=used + scale_loads,
        vb_one_pass_bf16=2 * used,
        vb_one_pass_fp8=used + scale_loads,
        hb_grad_bf16=grad * 2,
        hb_grad_fp8=grad + 4 * b * h * ((m + 31) // 32),
        hb_state_prev_bf16=4 * b * h * (w - 1) * m,
        hb_state_prev_fp8=2 * b * h * (w - 1) * m + scales,
        saved_state_bytes=n - scales,
    )


def metrics(ref, val):
    a, b = ref.detach().double().flatten(), val.detach().double().flatten()
    return dict(
        cos=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        rel_l2=float((a - b).norm() / a.norm().clamp_min(1e-30)),
    )


def horizontal(x, a, g, previous=None):
    out = []
    for reverse in (False, True):
        z = torch.zeros_like(x[:, :, 0])
        row = []
        coeff = a.conj() if reverse else a
        gain = g.conj() if reverse else g
        for t in range(x.shape[2]):
            k = x.shape[2] - 1 - t if reverse else t
            before = z
            z = coeff * z + gain * x[:, :, k]
            if previous is not None and t:
                prev = previous[int(reverse)][:, :, k + (1 if reverse else -1)]
                z = z + (coeff - coeff.detach()) * (prev - before).detach()
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


def simulate(h, w, m, c):
    torch.manual_seed(617)
    x = (torch.randn(1, h, w, m, dtype=torch.complex64) * 0.1).requires_grad_()
    a = torch.full((m,), 0.73 + 0.13j, requires_grad=True)
    g = torch.full((m,), 0.21 + 0.09j, requires_grad=True)
    st = horizontal(x, a, g)
    planes = torch.stack((st[0].real, st[0].imag, st[1].real, st[1].imag)).detach()
    chunks = []
    for group in planes.split(32, -1):
        scale = group.abs().amax((0, 3, 4), keepdim=True).clamp_min(1e-12) / 448
        chunks.append((group / scale).to(torch.float8_e4m3fn).float() * scale)
    packed = torch.cat(chunks, -1)
    qs = [torch.complex(packed[0], packed[1]), torch.complex(packed[2], packed[3])]
    aux = [torch.rand(2, s, m) + 1 for s in (w, h)]
    path = [torch.randn(s) * 0.125 for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    grad = torch.randn_like(planes)
    if c:
        grad[:2, :, :, ::2] = 0
        grad[2:, :, :, 1::2] = 0
        if w % 2:
            grad[2:, :, :, -1] = 0
    scale = grad.abs().amax().clamp_min(1e-12) / 57344
    dq = (grad / scale).to(torch.float8_e5m2).float() * scale

    def gradients(s, d):
        loss = sum((z.real * d[i * 2] + z.imag * d[i * 2 + 1]).sum() for i, z in enumerate(s))
        return torch.autograd.grad(loss, (x, a, g))

    ref = gradients(st, grad)
    test = gradients(horizontal(x, a, g, qs), dq)
    result = {
        "states": metrics(planes, packed),
        "d_states": metrics(grad, dq),
        "output": metrics(vertical(st, a, g, aux, path, c), vertical(qs, a, g, aux, path, c)),
    }
    for key, r, t in zip(("dx", "da", "dg"), ref, test, strict=True):
        result[key] = metrics(torch.view_as_real(r), torch.view_as_real(t))
    return result


def offline():
    result = {}
    torch.set_num_threads(1)
    for label, h, w, m, c, count in [*MODEL_SHAPES, *EDGE]:
        res = {} if (label.startswith("odd")) else baseline(h, w, m, c)
        for name, entry, config, threads in [
            ("A", "vf", (h, w, c), 128),
            ("A_fast", "vf", (h, w, c, False, True), 128),
            ("B_vf", "vf", (h, w, c, True), 128),
            ("B_hf", "hf", (h, w), 32),
            ("C", "hb", (h, w, c), 32),
            ("quant", "quant", (1, 1), 256),
        ]:
            row = resources(fp.image(*config), entry, threads)
            if any(row[k] for k in ("local", "stack", "spill_loads", "spill_stores")):
                raise RuntimeError((label, name, row))
            res[name] = row
        result[label] = dict(
            resources=res,
            traffic=traffic(1024, h, w, m, c),
            count=count,
            precision=simulate(h, w, m, c),
        )
    return result


def device_precision(h, w, m, c):
    """Compare real CUDA entry points with independent FP32 recurrences at B=1."""
    torch.manual_seed(615)
    x = torch.randn(1, h, w, 2 * m, dtype=torch.bfloat16) * 0.125
    r, i = x.split(m, -1)
    z = torch.complex(r.float(), i.float()).requires_grad_()
    a = torch.full((m,), 0.73 + 0.13j, requires_grad=True)
    g = torch.full((m,), 0.21 + 0.09j, requires_grad=True)
    pole = [p.detach().contiguous().cuda() for p in (a.real, a.imag, g.real, g.imag)]
    aux = [1 + torch.rand(2, s, m) for s in (w, h)]
    path = [(torch.randn(s) * 0.125).bfloat16() for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    st = horizontal(z, a, g)
    planes = torch.stack([st[0].real, st[0].imag, st[1].real, st[1].imag])
    ref = vertical(st, a, g, aux, [p.float() for p in path], c)
    d = torch.randn(4, 1, h, w // (1 + c), m).bfloat16()
    if c:
        loss = (planes[:2, :, :, 1::2] * d[:2]).sum() + (
            planes[2:, :, :, 0 : 2 * (w // 2) : 2] * d[2:]
        ).sum()
    else:
        loss = (planes * d).sum()
    dx, da, dg = torch.autograd.grad(loss, (z, a, g))
    rx, ix = x.cuda().split(m, -1)
    aux = [p.cuda() for p in aux]
    path = [p.cuda() for p in path]
    bf = planes.detach().bfloat16().cuda()
    adj = d.cuda()

    def run():
        q, s = fp.w61_hscan_fwd_fp8_out(pole, rx, ix)
        dq, ds = fp.quantize_d_states(adj)
        gradients = fp.w61_hscan_bwd_fp8_in(pole, rx, ix, q, s, dq, ds)
        return (
            q,
            s,
            fp.w61_vscan_fwd_fp8_smem(pole, bf, aux, path, c),
            fp.w61_vscan_fwd_fp8_smem_fastsilu(pole, bf, aux, path, c),
            fp.w61_vscan_fwd_fp8_in(pole, q, s, aux, path, c),
            *gradients,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        vals = run()
    graph.replay()
    q, s, *outs = [t.cpu() for t in vals]
    decoded = q.view(torch.float8_e4m3fn).float() * s.repeat_interleave(32, -1)[..., :m].unsqueeze(
        2
    )
    return {
        name: metrics(r, t)
        for name, r, t in zip(
            ("hf", "A", "A_fast", "B", "da_r", "da_i", "dg_r", "dg_i", "dx"),
            (
                planes,
                ref,
                ref,
                ref,
                da.real,
                da.imag,
                dg.real,
                dg.imag,
                torch.cat([dx.real, dx.imag], -1),
            ),
            (decoded, *outs),
            strict=True,
        )
    }


def benchmark(b, warmup, iters):
    os.environ.update(lockin_arm_env("w57_lockin"))
    result = {}
    totals = {}
    for label, h, w, m, c, count in MODEL_SHAPES:
        inp = make_inputs(b, h, w, m, c, torch.bfloat16)
        pole, _, aux, path, *_ = inp
        x = torch.randn(b, h, w, 2 * m, device="cuda", dtype=torch.bfloat16)
        r, i = x.split(m, -1)
        hf = lambda: scans.horizontal_forward(pole, r, i)
        bf = lambda: fp.w61_hscan_fwd_fp8_out(pole, r, i)
        states = hf()
        packed = bf()
        vf = lambda st: scans.vertical_forward_packed(pole, st, aux, path, c, 1e-6, planar=False)
        bv = lambda pair: fp.w61_vscan_fwd_fp8_in(pole, *pair, aux, path, c)
        grad = torch.randn((4, b, h, w // (1 + c), m), device="cuda", dtype=r.dtype)
        dq = fp.quantize_d_states(grad)
        ops = {
            "w57_hf": hf,
            "B_hf": bf,
            "w57_vf": lambda: vf(states),
            "A": lambda: fp.w61_vscan_fwd_fp8_smem(pole, states, aux, path, c),
            "A_fast": lambda: fp.w61_vscan_fwd_fp8_smem_fastsilu(pole, states, aux, path, c),
            "B_vf": lambda: bv(packed),
            "w57_pipeline": lambda: vf(hf()),
            "B_pipeline": lambda: bv(bf()),
            "w57_hb": lambda: scans.horizontal_backward(pole, [r, i], states, grad, compact=c),
            "C": lambda: fp.w61_hscan_bwd_fp8_in(pole, r, i, *packed, *dq),
        }
        result[label] = {}
        for name, op in ops.items():
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
            outputs = out if isinstance(out, (tuple, list)) else [out]
            finite = all(torch.isfinite(y).all().item() for y in outputs)
            del graph, out, outputs
            ms = _time(op, warmup, iters)
            result[label][name] = dict(ms=ms, peak_bytes=peak, finite=finite)
            totals[name] = totals.get(name, 0) + count * ms
        result[label]["pipeline_peak_saved_bytes"] = (
            result[label]["w57_pipeline"]["peak_bytes"] - result[label]["B_pipeline"]["peak_bytes"]
        )
        result[label]["precision_fp32"] = device_precision(h, w, m, c)
    return dict(shapes=result, weighted_ms=totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-audit", action="store_true")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = offline() if args.offline_audit else benchmark(args.batch, args.warmup, args.iters)
    data = json.dumps(result, separators=(",", ":"), allow_nan=False)
    if args.json:
        args.json.write_text(data + "\n")
    if args.offline_audit:
        print("shape kernel regs smem_B threads AW/SM stack local spill_loads spill_stores")
        for label, shape in result.items():
            for kernel, r in shape["resources"].items():
                print(
                    label,
                    kernel,
                    *(
                        r[k]
                        for k in (
                            "regs",
                            "smem",
                            "threads",
                            "aw_sm",
                            "stack",
                            "local",
                            "spill_loads",
                            "spill_stores",
                        )
                    ),
                )
            print(label, "CPU precision", shape["precision"])
    else:
        print(data)
