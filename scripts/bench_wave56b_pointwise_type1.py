"""Time Wave 56B pointwise operators for the 224px VA-Ablation-17 workload."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from lnet import model_w29c as ops

MODES = ("off", "triton_hiocc", "cuda_vec8", "cuda_vec8_tanh")
# rows per image, hidden/width, input pitch, incoming gradient pitch, calls
SWIGLU = (
    (3136, 96, 192, 96, 1),
    (784, 144, 288, 144, 2),
    (196, 192, 384, 192, 6),
    (49, 384, 768, 384, 2),
)
RMS = (
    (3136, 64, 64, 64, 3),
    (784, 96, 96, 96, 4),
    (196, 128, 128, 128, 12),
    (49, 256, 256, 256, 4),
)


def case(kind, shape, batch, dtype, device):
    per_image, width, xp, gp, _ = shape
    rows = batch * per_image

    def sample(pitch):
        return torch.randn(rows, pitch, device=device, dtype=dtype)

    if kind == "rms":
        x = sample(xp)[:, :width]
        dy = sample(gp)[:, :width]
        w = torch.randn(width, device=device, dtype=dtype)
        inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)

        def run():
            return ops._channel_rmsnorm_backward(x, w, inv, dy, need_dx=True, need_dw=True)

        def ref():
            n = x.float() * inv
            s = dy.float() * w.float()
            return (
                (s - (n / width) * (s * n).sum(-1, keepdim=True)) * inv,
                (dy.float() * n).sum(0),
            )

        traffic = (3 * rows * width + 2 * width) * x.element_size() + 4 * rows
    else:
        x = sample(xp)
        dy = sample(gp)

        def run():
            y = ops.packed_swiglu_vec2d(x) if kind == "fwd" else ops._swiglu_vec2d_backward(x, dy)
            return (y,)

        def ref():
            a, b = x.float().chunk(2, -1)
            if kind == "fwd":
                return (a * F.silu(b),)
            sig = b.sigmoid()
            return (torch.cat((dy * F.silu(b), dy * a * sig * (1 + b * (1 - sig))), -1),)

        traffic = (3 if kind == "fwd" else 5) * rows * width * x.element_size()
    return run, ref, traffic


def errors(actual, reference):
    return [
        ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)).item()
        for x, y in zip(actual, reference, strict=True)
    ]


def screen(args):
    if args.batch <= 0 or args.iters <= 0:
        raise ValueError("batch and iters must be positive")
    cache = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    results = []
    totals = dict.fromkeys(args.modes, 0.0)
    for kind, shapes in (("fwd", SWIGLU), ("bwd", SWIGLU), ("rms", RMS)):
        for shape in shapes:
            for dtype, tol in ((torch.bfloat16, 0.01), (torch.float16, 0.003)):
                run, ref, traffic = case(kind, shape, args.batch, dtype, "cuda")
                with patch.dict(os.environ, LNET_MODEL_W56B="off"):
                    baseline = run()
                reference = ref()
                for mode in args.modes:
                    with patch.dict(os.environ, LNET_MODEL_W56B=mode):
                        actual = run()
                        vs_ref, vs_off = errors(actual, reference), errors(actual, baseline)
                        assert max(vs_ref + vs_off) <= tol, (kind, shape, mode, vs_ref, vs_off)
                        torch.compiler.reset()
                        compiled = torch.compile(run, fullgraph=True)
                        assert max(errors(compiled(), actual)) <= tol
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = run()
                        graph.replay()
                        assert all(t.isfinite().all().item() for t in captured)
                        del compiled, actual
                        ms = None
                        if dtype == torch.bfloat16:
                            for _ in range(3):
                                graph.replay()
                            samples = []
                            for _ in range(args.iters):
                                cache.add_(1)
                                start, end = (
                                    torch.cuda.Event(enable_timing=True) for _ in range(2)
                                )
                                start.record()
                                graph.replay()
                                end.record()
                                end.synchronize()
                                samples.append(start.elapsed_time(end))
                            ms = sum(samples) / len(samples)
                            totals[mode] += ms * shape[-1]
                        del graph, captured
                        # Operator payload plus explicit FP32 CTA subtotal writes and reads.
                        # These are algorithmic bytes. Actual HBM traffic needs counters.
                        extra = 0
                        if kind == "rms":
                            rows, width = shape[0] * args.batch, shape[1]
                            sms = torch.cuda.get_device_properties(0).multi_processor_count
                            if mode.startswith("cuda_vec8"):
                                blocks = min(
                                    sms * (4 if mode.endswith("tanh") else 8), (rows + 7) // 8
                                )
                                if width <= 256:
                                    extra += (
                                        2
                                        * rows
                                        * width
                                        * torch.empty((), dtype=dtype).element_size()
                                        + 4 * rows
                                    )
                            else:
                                tile = (
                                    8
                                    if mode == "triton_hiocc"
                                    else ops._rmsnorm_backward_rows(width)
                                )
                                blocks = min(
                                    sms * (6 if mode == "triton_hiocc" else 2),
                                    (rows + tile - 1) // tile,
                                )
                            extra += blocks * width * 8
                        results.append(
                            dict(
                                kind=kind,
                                shape=shape,
                                mode=mode,
                                dtype=str(dtype),
                                ms=ms,
                                logical_bytes=traffic,
                                algorithmic_bytes=traffic + extra,
                                algorithmic_GBs=(traffic + extra) / ms / 1e6 if ms else None,
                                logical_GBs=traffic / ms / 1e6 if ms else None,
                                rel_l2_fp32=vs_ref,
                                rel_l2_off=vs_off,
                                graph_finite=True,
                                fullgraph=True,
                            )
                        )
                del run, ref, baseline, reference
    off = {
        (r["kind"], r["shape"]): r["ms"]
        for r in results
        if r["mode"] == "off" and r["ms"] is not None
    }
    for r in results:
        key = (r["kind"], r["shape"])
        if r["ms"] is not None:
            r["weighted_ms"] = r["ms"] * r["shape"][-1]
            r["delta_ms"] = r["ms"] - off[key] if key in off else None
    return dict(
        batch=args.batch,
        shapes=results,
        weighted_ms=totals,
        delta_ms={k: v - totals["off"] for k, v in totals.items()} if "off" in totals else {},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.selftest:
        assert sum(s[-1] for s in SWIGLU) * 2 == 22 and sum(s[-1] for s in RMS) == 23
        for kind in ("fwd", "bwd", "rms"):
            _, ref, _ = case(
                kind, (3, 8, 8 if kind == "rms" else 16, 8, 1), 1, torch.bfloat16, "cpu"
            )
            assert all(t.isfinite().all() for t in ref())
        if torch.cuda.is_available():
            for mode in args.modes:
                with patch.dict(os.environ, LNET_MODEL_W56B=mode):
                    for kind in ("fwd", "bwd", "rms"):
                        run, ref, _ = case(
                            kind, (3, 8, 8 if kind == "rms" else 16, 8, 1), 1, torch.float16, "cuda"
                        )
                        assert max(errors(run(), ref())) <= 0.003
        result = {"selftest": "PASS"}
    else:
        result = screen(args)
    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
