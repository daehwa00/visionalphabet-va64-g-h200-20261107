"""Measure the Wave 17B, native ATen, and custom RMSNorm candidates."""
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lnet.fused_rmsnorm_ops import fused_rmsnorm
from lnet.memory_backbone import ChannelRMSNorm, _CompiledChannelRMSNorm

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
import bench_h200 as baseline


def shapes(batch: int) -> list[dict]:
    from torch import nn

    model = baseline.build_model(baseline.VARIANTS[0])
    result = []

    def hook(name: str, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        result.append({
            "name": name, "shape": list(x.shape), "stride": x.stride(),
            "dtype": str(x.dtype), "weight_dtype": str(module.weight.dtype),
            "eps": module.epsilon if isinstance(module, ChannelRMSNorm) else module.eps,
        })

    for name, module in model.named_modules():
        if isinstance(module, (ChannelRMSNorm, nn.RMSNorm)):
            module.register_forward_pre_hook(lambda m, x, name=name: hook(name, m, x))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(baseline.setup(batch))
    return result


def graph_time(fn: Callable, repeats: int = 15) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(12):
            fn()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 12)
    return statistics.median(samples)


def isolated(batch: int, out: Path, shard: int) -> None:
    calls = shapes(2)
    cases = {
        (batch * (torch.Size(r["shape"]).numel() // (2 * r["shape"][-1])),
         r["shape"][-1], r["eps"])
        for r in calls if r["shape"][0] == 2 and len(r["shape"]) > 1
    }
    cases.update((1027, c, 1e-6) for c in (47, 64, 160, 320, 640, 1024))
    eps = 1e-6

    def native(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)

    def old(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return _CompiledChannelRMSNorm.apply(x, w, eps)

    def custom(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, w, eps)

    with out.open("w") as log:
        log.write(json.dumps({"model_layouts_b2": calls}) + "\n")
        for n, c, eps in sorted(cases)[shard::3]:
            torch.compiler.reset()
            for dtype in (torch.bfloat16, torch.float16):
                x = torch.randn(n, c, device="cuda", dtype=dtype, requires_grad=True)
                w = torch.randn(c, device="cuda", dtype=dtype, requires_grad=True)
                dy = torch.randn_like(x)
                for name, fn in (("off", old), ("native", native), ("on", custom)):
                    compiled = torch.compile(fn, fullgraph=True) if name != "on" else fn
                    # Retain the forward graph so backward timing excludes the forward launch.
                    y = compiled(x, w)

                    forward_ms = graph_time(lambda fn=compiled, x=x, w=w: fn(x, w))
                    backward_ms = graph_time(
                        lambda y=y, x=x, w=w, dy=dy: torch.autograd.grad(
                            y, (x, w), dy, retain_graph=True
                        )
                    )
                    bytes_x = x.numel() * x.element_size()
                    log.write(json.dumps({
                        "n": n, "c": c, "eps": eps, "dtype": str(dtype), "arm": name,
                        "forward_ms": forward_ms, "backward_ms": backward_ms,
                        "total_ms": forward_ms + backward_ms,
                        "forward_effective_gbs": 2 * bytes_x / forward_ms / 1e6,
                        "backward_effective_gbs": 3 * bytes_x / backward_ms / 1e6,
                    }) + "\n")
                    log.flush()
                    del y


def profile_step(step: Callable) -> dict:
    from collections import Counter

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        step()
        torch.cuda.synchronize()
    times, counts = Counter(), Counter()
    kernels = []
    for event in profile.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = event.key
        bucket = "custom_rmsnorm" if name.startswith("rms_") else baseline.kernel_bucket(name)
        ms = event.self_device_time_total / 1000
        times[bucket] += ms
        counts[bucket] += event.count
        kernels.append({"name": name, "count": event.count, "self_cuda_ms": ms})
    times["total"] = sum(times.values())
    return {
        "self_cuda_ms_by_bucket": times,
        "dispatches_by_profiler_bucket": counts,
        "kernels": sorted(kernels, key=lambda row: -row["self_cuda_ms"]),
    }


def train(args: argparse.Namespace) -> dict:
    setup = baseline.setup

    def configure(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = args.cudnn == "true"
        return inputs

    measure = baseline.measure
    trajectory = []

    def measure_losses(
        step: Callable, batch: int, *, warmup: int, count: int
    ) -> tuple[dict, list]:
        result, losses = measure(step, batch, warmup=warmup, count=count)
        trajectory.extend(torch.stack(losses).tolist())
        return result, losses

    baseline.setup = configure
    baseline.measure = measure_losses
    baseline.profile_step = profile_step
    result = baseline.train_arm(args)
    result["measured_loss_trajectory"] = trajectory
    result["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    for key in ("peak_allocated", "peak_reserved"):
        result[f"{key}_bytes"] = round(result[f"{key}_gib"] * baseline.GIB)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("train", "isolated"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--cudnn", choices=("true", "false"), default="true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(3), default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.variant = baseline.VARIANTS[0]
    if args.kind == "isolated":
        isolated(args.batch, args.out, args.shard)
    else:
        baseline.write_json(args.out, train(args))


if __name__ == "__main__":
    main()
