"""Wave 62B standalone audit and H200 screening."""

import argparse
import ast
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
from lnet import cudnn_fp8_conv as ops


def metrics(a, b):
    sums = torch.zeros(4, device=a.device, dtype=torch.float64)
    for u, v in zip(a.detach().reshape(-1).split(1 << 20), b.detach().reshape(-1).split(1 << 20)):
        u, v = u.float(), v.float()
        sums += torch.stack(
            ((u - v).square().sum(), v.square().sum(), u.square().sum(), (u * v).sum())
        )
    err, rr, aa, dot = sums.tolist()
    return dict(cosine=dot / math.sqrt(max(aa * rr, 1e-30)), rel_l2=math.sqrt(err / max(rr, 1e-30)))


def inputs(h, c, k, batch, device):
    def make(shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16).contiguous(
            memory_format=ops.CL
        )

    return make((batch, c, h, h)), make((k, c, 3, 3)), make((batch, k, h, h))


def bfops(x, w, g):
    return (
        lambda: F.conv2d(x, w, padding=1),
        lambda: torch.nn.grad.conv2d_input(x.shape, w, g, padding=1),
        lambda: torch.nn.grad.conv2d_weight(x, w.shape, g, padding=1),
    )


def parity(h, c, k):
    x, w, g = inputs(h, c, k, 1, "cpu")

    def q(t, dtype):
        s = t.float().abs().amax() / torch.finfo(dtype).max
        return (t.float() / s).to(dtype).float(), s

    a, sa = q(x, ops.E4)
    b, sb = q(w, ops.E4)
    d, sd = q(g, ops.E5)
    return {
        name: metrics((f() * s).bfloat16(), ref())
        for name, f, s, ref in zip(
            ops.OPS,
            bfops(a, b, d),
            (sa * sb, sd * sb, sa * sd),
            bfops(x.float(), w.float(), g.float()),
        )
    }


def compile_kernels():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    target = GPUTarget("cuda", 90, 32)
    result = {}

    def compile_one(fn, types, constants, name):
        sig = dict(zip(fn.arg_names, types))
        sig.update({key: "constexpr" for key in constants})
        kernel = triton.compile(ASTSource(fn, sig, constants), target=target)
        data = kernel.asm["cubin"]
        assert data.startswith(b"\x7fELF")
        result[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    sizes = dict(N=1024 * 56 * 56 * 320, BLOCK=1024)
    compile_one(ops.triton_amax, ["*bf16", "*i32"], sizes, "amax")
    for typ, limit in (("fp8e4nv", 448.0), ("fp8e5", 57344.0)):
        for track in (True, False):
            compile_one(
                ops.triton_cast_fp8_and_amax,
                ["*bf16", "*" + typ, "*fp32", "*i32"],
                dict(**sizes, LIMIT=limit, TRACK=track),
                f"{typ}_track_{track}",
            )
        compile_one(ops._update, ["*i32", "*fp32", "*fp32"], dict(LIMIT=limit), "update_" + typ)
    for typ in ("bf16", "fp32"):
        compile_one(ops._scale, ["*" + typ, "*fp32", "*bf16"], sizes, "scale_" + typ)
    return result


def hot_check():
    for fn in (
        ops._Conv.forward,
        ops._Conv.backward,
        ops.Runner.quant,
        ops.Runner.op,
        ops.Device.workspace,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        assert not any(
            isinstance(n, ast.Attribute) and n.attr in {"item", "cpu", "synchronize"}
            for n in ast.walk(tree)
        )


def audit():
    assert not torch.cuda.is_available()
    torch.manual_seed(225)
    hot_check()
    result = {}
    for h, c, k, _ in ops.SHAPES:
        nhwc = torch.empty((1024, h, h, c), device="meta")
        view = nhwc.permute(0, 3, 1, 2)
        assert view._base is nhwc and view.is_contiguous(memory_format=ops.CL)
        result[str(h)] = parity(h, c, k)
        assert min(v["cosine"] for v in result[str(h)].values()) >= 0.995
    return dict(
        batch_simulated=1,
        production_batch_layout=1024,
        parity=result,
        hot_check=True,
        sm90=compile_kernels(),
    )


def measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters


def screen(spec, warmup, iters):
    from lnet.memory_backbone import dense_nhwc_conv2d, dense_nhwc_conv2d_backward

    h, c, k, calls = spec
    x, w, g = inputs(h, c, k, 1024, "cuda")
    nh = lambda t: t.permute(0, 2, 3, 1)
    xx = nh(x).detach().requires_grad_()
    w.requires_grad_()
    grad = nh(g)
    ref = [fn().detach() for fn in bfops(x.float(), w.detach().float(), g.float())]
    rows = {}
    capture_stream = torch.cuda.Stream()

    def evaluate(name, forward, parts, quant=None, runner=None):
        def run():
            y = forward(xx, w)
            return (y, *torch.autograd.grad(y, (xx, w), grad))

        row = {op + "_ms": measure(fn, warmup, iters) for op, fn in zip(ops.OPS, parts)}
        row["quant_ms"] = measure(quant, warmup, iters) if quant else 0.0
        row["total_ms"] = measure(run, warmup, iters)
        vals = run()
        row["accuracy"] = {
            op: metrics(v, r) for op, v, r in zip(ops.OPS, vals, (nh(ref[0]), nh(ref[1]), ref[2]))
        }
        del vals
        graph = torch.cuda.CUDAGraph()
        stream = capture_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            vals = run()
        graph.replay()
        row["graph_finite"] = all(v.isfinite().all().item() for v in vals)
        if runner:
            row["plans"] = dict(runner.info)
        rows[name] = row

    base = bfops(x, w, g)
    dense = lambda a, b: dense_nhwc_conv2d(a, b, 1)
    evaluate(
        "bf16_dense_nhwc",
        dense,
        (
            lambda: dense(xx, w),
            lambda: dense_nhwc_conv2d_backward(grad, xx, w, 1, [True, False]),
            lambda: dense_nhwc_conv2d_backward(grad, xx, w, 1, [False, True]),
        ),
    )
    evaluate(
        "bf16_functional", lambda a, b: nh(F.conv2d(a.permute(0, 3, 1, 2), b, padding=1)), base
    )
    for mode in ("delayed", "dynamic"):
        for enabled in (ops.OPS, *[(op,) for op in ops.OPS]):
            runner = ops.Runner(mode, enabled)

            def quant():
                return (
                    runner.quant(x, "x") if set(enabled) & {"fprop", "wgrad"} else None,
                    runner.quant(w, "w") if set(enabled) & {"fprop", "dgrad"} else None,
                    runner.quant(g, "dy") if set(enabled) & {"dgrad", "wgrad"} else None,
                )

            a, b, d = quant()
            pairs = ((a, b, g.shape), (d, b, x.shape), (a, d, w.shape))
            parts = [
                (lambda op=op, p=p: runner.op(op, *p)) if op in enabled else f
                for op, p, f in zip(ops.OPS, pairs, base)
            ]
            label = "cudnn_fp8_" + mode + ("" if enabled == ops.OPS else "_" + enabled[0] + "_only")
            evaluate(label, runner, parts, quant, runner)
        # Include all remaining mixed choices so hybrid selection includes shared quantization costs.
        for omitted in ops.OPS:
            enabled = tuple(op for op in ops.OPS if op != omitted)
            runner = ops.Runner(mode, enabled)
            a, b, d = [runner.quant(t, s) for t, s in zip((x, w, g), ("x", "w", "dy"))]
            pairs = ((a, b, g.shape), (d, b, x.shape), (a, d, w.shape))
            parts = [
                (lambda op=op, p=p: runner.op(op, *p)) if op in enabled else f
                for op, p, f in zip(ops.OPS, pairs, base)
            ]
            evaluate(
                mode + "_" + "_".join(enabled),
                runner,
                parts,
                lambda: [runner.quant(t, s) for t, s in zip((x, w, g), ("x", "w", "dy"))],
                runner,
            )
    best = min((s for s in rows if rows[s]["graph_finite"]), key=lambda s: rows[s]["total_ms"])
    rows["cudnn_fp8_hybrid_best"] = dict(rows[best], selected=best)
    return dict(calls=calls, rows=rows)


def train_gate():
    from run_va_eight_ablations import build_variant
    from lnet import memory_backbone
    from lnet.model_w36b import prepare_model_w36b_
    from h200.bench_h200 import LOCKIN_ENV

    original = memory_backbone.dense_nhwc_conv2d
    runners = {}
    hits = 0

    def conv(x, w, p):
        nonlocal hits
        if p == 1 and (x.shape[1], x.shape[-1], w.shape[0]) in [s[:3] for s in ops.SHAPES]:
            hits += 1
            # Direct caller is the layer's _dense_nhwc method.
            layer = sys._getframe(1).f_locals["self"]
            if layer not in runners:
                runners[layer] = ops.Runner()
            return runners[layer](x, w)
        return original(x, w, p)

    with patch.dict(os.environ, LOCKIN_ENV):
        torch.manual_seed(225)
        model = prepare_model_w36b_(
            build_variant("17-p1-nr4-1024d-v3").cuda().to(memory_format=ops.CL)
        )
        models = (model, deepcopy(model))
        opts = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in models]
        steps = []
        for i in range(10):
            x = torch.randn(16, 3, 224, 224, device="cuda").contiguous(memory_format=ops.CL)
            label = torch.randint(100, (16,), device="cuda")
            losses, grads = [], []
            for j, (m, opt) in enumerate(zip(models, opts)):
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(i)
                with (
                    patch.object(memory_backbone, "dense_nhwc_conv2d", conv) if j else nullcontext()
                ):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = F.cross_entropy(m(x)[0], label)
                    loss.backward()
                losses.append(loss.item())
                grads.append(
                    torch.cat(
                        [p.grad.float().flatten() for p in m.parameters() if p.grad is not None]
                    )
                )
                opt.step()
            steps.append(
                dict(
                    loss=losses,
                    drift=abs(losses[0] - losses[1]) / max(losses[0], 1e-12),
                    grad_cosine=metrics(*grads)["cosine"],
                )
            )
    return dict(
        steps=steps,
        fp8_calls=hits,
        passed=hits > 0 and all(s["drift"] <= 0.01 and s["grad_cosine"] >= 0.95 for s in steps),
    )


def compact(obj):
    if isinstance(obj, float):
        return float(f"{obj:.5g}")
    if isinstance(obj, dict):
        return {k: compact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return list(map(compact, obj))
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-audit", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.offline_audit:
        result = audit()
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        rows = {str(s[0]): screen(s, args.warmup, args.iters) for s in ops.SHAPES}
        totals = {
            n: sum(r["calls"] * r["rows"][n]["total_ms"] for r in rows.values())
            for n in next(iter(rows.values()))["rows"]
        }
        base = totals["bf16_dense_nhwc"]
        result = dict(
            shapes=rows,
            weighted_ms=totals,
            weighted_delta_ms={n: t - base for n, t in totals.items()},
            projected_delta_on_26_02_ms={n: 26.02 * (t / base - 1) for n, t in totals.items()},
            training=train_gate(),
        )
    doc = json.dumps(compact(result), allow_nan=False)
    if args.json:
        args.json.write_text(doc + "\n")
    print(doc)
    if not args.offline_audit and (
        not result["training"]["passed"]
        or not all(v["graph_finite"] for s in rows.values() for v in s["rows"].values())
    ):
        raise SystemExit("GPU validation failed")


if __name__ == "__main__":
    main()
