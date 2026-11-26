"""Whole-Hopper FP8 screening and CPU-only production shape audit."""

import argparse
import json
import os
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_va_eight_ablations import build_variant

VARIANT = "17-p1-nr4-1024d-v3"


def production_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from h200.bench_h200 import LOCKIN_ENV
    from lnet.model_w36b import prepare_model_w36b_

    conv, gemm = Counter(), Counter()

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.lnet.dense_nhwc_conv2d.default:
                x, w, pad = args
                conv[(*x.shape, w.shape[0], *w.shape[2:], pad)] += 1
            if func in (
                torch.ops.aten.mm.default,
                torch.ops.aten.addmm.default,
                torch.ops.aten.bmm.default,
            ):
                a, b = args[1:3] if func == torch.ops.aten.addmm.default else args[:2]
                gemm[(*a.shape, b.shape[-1])] += 1
            return func(*args, **(kwargs or {}))

    with patch.dict(os.environ, LOCKIN_ENV):
        model = prepare_model_w36b_(build_variant(VARIANT))
        with FakeTensorMode(allow_non_fake_inputs=True), torch.no_grad(), Record():
            model(torch.zeros(1024, 3, 224, 224))
    return [list(conv.items()), list(gemm.items())]


CONV = [
    ((1024, 56, 56, 64, 320, 3, 3, 1), 2),
    ((1024, 28, 28, 96, 320, 3, 3, 1), 2),
    ((1024, 14, 14, 128, 320, 3, 3, 1), 6),
    ((1024, 7, 7, 256, 256, 3, 3, 1), 2),
]

GEMM = [
    ((3211264, 64, 64), 3),
    ((64, 64, 64), 1),
    ((160, 18, 4, 64), 2),
    ((3211264, 160, 64), 2),
    ((3211264, 64, 192), 1),
    ((3211264, 96, 64), 1),
    ((802816, 160, 96), 4),
    ((802816, 64, 96), 1),
    ((802816, 96, 288), 2),
    ((802816, 144, 96), 2),
    ((160, 18, 4, 96), 2),
    ((200704, 160, 128), 12),
    ((200704, 96, 128), 1),
    ((200704, 128, 384), 6),
    ((200704, 192, 128), 6),
    ((160, 18, 4, 128), 6),
    ((50176, 160, 256), 2),
    ((50176, 128, 256), 1),
    ((50176, 256, 768), 2),
    ((50176, 384, 256), 2),
    ((128, 18, 4, 256), 2),
    ((128, 7168, 7, 7), 8),
    ((1024, 1024, 100), 1),
]

from lnet import hopper_fp8_ops as ops
from lnet.memory_backbone import dense_nhwc_conv2d


def metrics(actual, ref):
    # Stream reductions so full production activations fit alongside the reference.
    sums = torch.zeros(4, device=actual.device, dtype=torch.float64)
    for a, b in zip(
        actual.detach().reshape(-1).split(1 << 20), ref.detach().reshape(-1).split(1 << 20)
    ):
        a, b = a.float(), b.float()
        sums += torch.stack(
            ((a - b).square().sum(), b.square().sum(), a.square().sum(), (a * b).sum())
        )
    err, rr, aa, dot = sums.tolist()
    return {"rel_l2": (err / max(rr, 1e-30)) ** 0.5, "cosine": dot / max((rr * aa) ** 0.5, 1e-30)}


def sample(shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)


def evaluate(fn, x, w, g, ref, reps):
    def run():
        y = fn(x, w)
        return y, *torch.autograd.grad(y, (x, w), g)

    vals = run()
    errors = {key: metrics(a, b) for key, a, b in zip(("out", "dx", "dw"), vals, ref)}
    del vals
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stamps = []
    for _ in range(reps):
        a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        a.record()
        y = fn(x, w)
        b.record()
        grads = torch.autograd.grad(y, (x, w), g)
        c.record()
        stamps.append((a, b, c))
        del grads, y
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    times = [
        sum(s[i].elapsed_time(s[j]) for s in stamps) / reps for i, j in ((0, 1), (1, 2), (0, 2))
    ]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        vals = run()
    graph.replay()
    return dict(
        zip(("fwd_ms", "bwd_ms", "total_ms"), times),
        errors=errors,
        peak_extra_MiB=peak / 2**20,
        graph_finite=all(v.isfinite().all().item() for v in vals),
    )


def screen(kind, shape, calls, reps, caps):
    if kind == "conv":
        b, h, width, c, n, kh, kw, pad = shape
        x, w = sample((b, h, width, c)), sample((n, c, kh, kw))
        w = w.detach().contiguous(memory_format=torch.channels_last).requires_grad_()
        baseline = lambda x, w: dense_nhwc_conv2d(x, w, pad)
        modes = {"bf16": baseline, "im2col": ops.fp8_dense_nhwc_conv2d}
        if caps["cudnn"]["available"]:
            modes["cudnn"] = partial(ops.fp8_dense_nhwc_conv2d, mode="cudnn")
    elif len(shape) == 4:
        groups, m, k, n = shape
        x, w = sample((groups, m, k)), sample((groups, n, k))
        baseline = lambda x, w: torch.bmm(x, w.transpose(1, 2))
        # Both production BMM families have unaligned K. Keep their grouped semantics.
        modes = {"bf16": baseline, "unaligned_fallback": baseline}
    else:
        m, k, n = shape
        x, w = sample((m, k)), sample((n, k))
        baseline = F.linear
        modes = {"bf16": baseline}
        for name in ("tensorwise", "rowwise"):
            if name == "rowwise" and not caps[name]["available"]:
                continue
            for fast in (False, True):
                for fused in (False, True):
                    modes[f"{name}_fast{int(fast)}_fused{int(fused)}"] = partial(
                        ops.fp8_linear, mode=name, fast=fast, fused=fused
                    )
    xf, wf = (t.detach().float().requires_grad_() for t in (x, w))
    y = baseline(xf, wf)
    g = torch.randn_like(y, dtype=x.dtype)
    ref = (y.detach(), *torch.autograd.grad(y, (xf, wf), g.float()))
    del y, xf, wf
    for compiled in (False, True) if kind == "gemm" else (False,):
        for mode, fn in modes.items():
            try:
                if compiled:
                    torch._dynamo.reset()
                record = evaluate(
                    torch.compile(fn, fullgraph=True) if compiled else fn, x, w, g, ref, reps
                )
            except Exception as exc:
                if mode != "cudnn":
                    raise
                yield dict(kind=kind, shape=shape, mode=mode, available=False, reason=str(exc))
                continue
            if mode == "bf16":
                baseline_times = {k: record[k] for k in ("fwd_ms", "bwd_ms", "total_ms")}
            delta = baseline_times["total_ms"] - record["total_ms"]
            yield dict(
                kind=kind,
                shape=shape,
                calls=calls,
                compiled=compiled,
                mode=mode,
                fwd_delta_ms=baseline_times["fwd_ms"] - record["fwd_ms"],
                bwd_delta_ms=baseline_times["bwd_ms"] - record["bwd_ms"],
                delta_ms=delta,
                weighted_delta_ms=calls * delta,
                **record,
            )


def train_gate(batch, steps):
    from copy import deepcopy
    from lnet.model_w36b import prepare_model_w36b_
    from h200.bench_h200 import LOCKIN_ENV

    with patch.dict(os.environ, LOCKIN_ENV):
        torch.manual_seed(219)
        a = prepare_model_w36b_(build_variant(VARIANT).cuda().to(memory_format=torch.channels_last))
        models = [a, ops.apply_fp8_module_patch(deepcopy(a))]
        opts = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in models]
        history = []
        for step in range(steps):
            x = torch.randn(batch, 3, 224, 224, device="cuda").contiguous(
                memory_format=torch.channels_last
            )
            label = torch.randint(100, (batch,), device="cuda")
            losses, grads = [], []
            for model, opt in zip(models, opts):
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(step)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x)[0], label)
                loss.backward()
                grads.append(
                    torch.cat(
                        [
                            p.grad.reshape(-1).float()
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                losses.append(loss.item())
                opt.step()
            history.append(
                dict(
                    step=step + 1,
                    bf16=losses[0],
                    fp8=losses[1],
                    relative_drift=abs(losses[1] - losses[0]) / max(losses[0], 1e-12),
                    gradient_cosine=metrics(*grads)["cosine"],
                )
            )
    return {"steps": history, "max_relative_drift": max(i["relative_drift"] for i in history)}


def audit_contracts():
    bounds = []
    for (batch, h, w, c, n, kh, kw, pad), calls in CONV:
        chunk = min(batch, ops.chunk_size(h, w, c, n))
        assert chunk > 0 and c % 16 == n % 16 == 0 and (kh, kw, pad) == (3, 3, 1)
        max_rows = (chunk * h * w + 15) // 16 * 16
        bound = max_rows * (18 * c + 8 * n + 4 * c) + 16 * 9 * c * n
        assert bound <= ops.LIMIT
        bounds.append(dict(hw=[h, w], chunk=chunk, max_scratch_MiB=bound / 2**20))
    with torch.device("meta"):
        for shape, _ in GEMM:
            m, k, n = shape[-3:]
            if len(shape) == 4 or m < 64 or any(dim % 16 for dim in (m, k, n)):
                continue
            for rows, reduction, cols in ((m, k, n), (m, n, k), (n, m, k)):
                lhs = torch.empty(rows, reduction, dtype=ops.E4)
                rhs = torch.empty(cols, reduction, dtype=ops.E4).T
                assert lhs.stride() == (reduction, 1) and rhs.stride() == (1, reduction)
    return bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-audit", action="store_true")
    parser.add_argument("--part", choices=("all", "conv", "gemm", "train"), default="all")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--batch", type=int, choices=(16, 32), default=16)
    args = parser.parse_args()
    if args.offline_audit:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
        assert production_shapes() == [CONV, GEMM]
        for (b, h, w, c, n, *_), _ in CONV:
            assert ops.chunk_size(h, w, c, n) > 0 and c % 16 == n % 16 == 0
        print(
            json.dumps(
                dict(
                    workspace=audit_contracts(),
                    variant=VARIANT,
                    conv=CONV,
                    gemm=GEMM,
                    capabilities={k: ops.probe(k) for k in ("tensorwise", "rowwise", "cudnn")},
                )
            )
        )
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error("Hopper required for screening")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    caps = {k: ops.probe(k) for k in ("tensorwise", "rowwise", "cudnn")}
    print(json.dumps(caps), flush=True)
    totals = Counter()
    for kind, table in (("conv", CONV), ("gemm", GEMM)):
        if args.part in ("all", kind):
            for shape, calls in table:
                for row in screen(kind, shape, calls, args.reps, caps):
                    print(json.dumps(row), flush=True)
                    if "weighted_delta_ms" in row:
                        totals[kind, row["mode"], row["compiled"]] += row["weighted_delta_ms"]
    for (kind, mode, compiled), value in totals.items():
        print(
            json.dumps(dict(kind=kind, mode=mode, compiled=compiled, total_weighted_delta_ms=value))
        )
    if args.part in ("all", "train"):
        gate = train_gate(args.batch, 10)
        print(json.dumps(gate))
        if any(
            not (i["gradient_cosine"] > 0 and i["relative_drift"] <= 0.1) for i in gate["steps"]
        ):
            raise SystemExit("training stability gate failed")


if __name__ == "__main__":
    main()
