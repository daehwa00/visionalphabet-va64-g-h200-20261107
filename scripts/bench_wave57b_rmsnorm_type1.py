"""Cold-cache RMSNorm backward timings on the VA-Ablation-17 shape mix."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from bench_wave56b_pointwise_type1 import RMS, case, errors
from lnet import model_w29c as ops


def screen(args):
    cache = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    results = []
    totals = dict.fromkeys(args.modes, 0.0)
    logical_total = 0
    for shape in RMS:
        for dtype, tol in ((torch.bfloat16, 0.01), (torch.float16, 0.003)):
            run, ref, traffic = case("rms", shape, args.batch, dtype, "cuda")
            if dtype == torch.bfloat16:
                logical_total += traffic * shape[-1]
            with patch.dict(os.environ, LNET_MODEL_W57B_RMS="off", LNET_MODEL_W56B="off"):
                baseline = run()
            reference = ref()
            for mode in args.modes:
                with patch.dict(os.environ, LNET_MODEL_W57B_RMS=mode, LNET_MODEL_W56B="off"):
                    actual = run()
                    vs_ref, vs_off = errors(actual, reference), errors(actual, baseline)
                    assert max(vs_ref + vs_off) <= tol, (shape, mode, vs_ref, vs_off)
                    torch.compiler.reset()
                    assert max(errors(torch.compile(run, fullgraph=True)(), actual)) <= tol
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run()
                    graph.replay()
                    assert all(t.isfinite().all().item() for t in captured)
                    ms = None
                    if dtype == torch.bfloat16:
                        for _ in range(3):
                            graph.replay()
                        times = []
                        for _ in range(args.iters):
                            cache.add_(1)
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            start.record()
                            graph.replay()
                            end.record()
                            end.synchronize()
                            times.append(start.elapsed_time(end))
                        ms = sum(times) / len(times)
                        totals[mode] += ms * shape[-1]
                    rows, width = args.batch * shape[0], shape[1]
                    if mode.startswith("cuda_sp"):
                        groups = 256 // ops._sp_group(width)
                        target = int(mode[-4])
                    else:
                        groups = ops._rmsnorm_backward_rows(width)
                        target = 2 if mode == "off" else 4
                    blocks = min(
                        target * ops._vec_sms(torch.cuda.current_device()),
                        (rows + groups - 1) // groups,
                    )
                    # The partial tensor is written once and consumed by sum(0).
                    results.append(
                        dict(
                            partial_read_write_bytes=8 * width * blocks,
                            input_read_amplification=1.0,
                            shape=shape,
                            mode=mode,
                            dtype=str(dtype),
                            ms=ms,
                            weighted_ms=ms * shape[-1] if ms else None,
                            logical_bytes=traffic,
                            logical_GBs=traffic / ms / 1e6 if ms else None,
                            algorithmic_payload_GBs=traffic / ms / 1e6 if ms else None,
                            rel_l2_fp32=vs_ref,
                            rel_l2_off=vs_off,
                            graph_finite=True,
                            fullgraph=True,
                        )
                    )
                    del graph, captured, actual
            del reference, baseline, run, ref
    off = {r["shape"]: r["ms"] for r in results if r["mode"] == "off" and r["ms"]}
    for r in results:
        if r["ms"] and r["shape"] in off:
            r["delta_ms"] = r["ms"] - off[r["shape"]]
    return dict(
        batch=args.batch,
        shapes=results,
        weighted_ms=totals,
        logical_GBs={m: logical_total / t / 1e6 for m, t in totals.items()},
        delta_ms={m: t - totals["off"] for m, t in totals.items()} if "off" in totals else {},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=ops._W57_MODES, default=ops._W57_MODES)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.iters) <= 0:
        parser.error("batch and iters must be positive")
    if args.selftest:
        assert sum(shape[-1] for shape in RMS) == 23
        for _, width, xp, gp, calls in RMS:
            _, ref, traffic = case("rms", (3, width, xp, gp, calls), 1, torch.bfloat16, "cpu")
            assert traffic == (9 * width + 2 * width) * 2 + 12
            assert all(t.isfinite().all() for t in ref())
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
