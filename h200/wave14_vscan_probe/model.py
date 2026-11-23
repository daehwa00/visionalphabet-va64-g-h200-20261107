"""Run matched P1/NR4 Wave 13 and fused-scan candidate comparisons."""

import argparse
import gc
import json
import os
import runpy
import statistics
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from variants import Arm, select

# ruff: noqa: SLF001, T201

SUITE = "h200/liverms_v2_opt/bench_h200.py"
P1 = "17-p1-nr4-1024d-v3"
ARMS = {
    "baseline": ("baseline", "baseline"),
    "compute": ("fast", "weights"),
    "cached": ("vector-line-source", "full-shared"),
    "line": ("line", "line"),
}


def graph_measure(
    step: Callable, batch: int, *, warmup: int, count: int,
) -> tuple[dict, list]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.graph(graph, stream=stream):
        output = step()
    graph.replay()
    torch.cuda.synchronize()
    measurements, outputs = [], []
    for _ in range(count):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(4):
            graph.replay()
        stop.record()
        stop.synchronize()
        measurements.append(start.elapsed_time(stop) / 4)
        outputs.append(output.clone())
    p25, median, p75 = statistics.quantiles(measurements, n=4, method="inclusive")
    return {
        "median_ms": median, "min_ms": min(measurements), "max_ms": max(measurements),
        "p25_ms": p25, "p75_ms": p75, "images_per_second": batch * 1000 / median,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "samples_ms": measurements,
    }, outputs


def annotate_payload(profile: dict, batch: int) -> None:
    for row in profile["scan_shape_cuda_events"]:
        row["mean_event_ms"] = row["cuda_event_ms"] / row["count"]
        if not row["vertical"]:
            continue
        b, h, w, m = batch, row["height"], row["width"], row["modes"]
        states = 4 * b * h * w * m * 2
        ratio = 1 + row["coarse"]
        outputs = 2 * b * (h // ratio) * (w // ratio) * m * 2
        minimum = states + outputs + (states // ratio if row["backward"] else 0)
        row["payload_bytes"] = minimum
        row["effective_GBps"] = minimum / (row["mean_event_ms"] * 1e6)


def checked(result: dict, command: str) -> bool:
    if result.get("status") != "ok":
        return False
    if command == "train-arm":
        flags = ("finite_losses", "finite_gradients", "running_ms_finite")
        return all(result[key] for key in flags)
    if command == "dynamo":
        graphs = [pair for rows in result["graphs"].values() for pair in rows.values()]
        return graphs == [[1, 0]] * 18
    return all((
        result["fp32_head_parity_by_precision"]["highest"]["passed"],
        result["strict_reload_passed"],
        result["folded_train_guard"],
    ))


def run(
    main: Callable, arm: Arm, command: str, out: Path, batch: int, *, profile: bool,
) -> bool:
    select(arm)
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    batch_options = [] if command == "dynamo" else ["--variant", P1, "--batch", str(batch)]
    profile_options = ["--profile"] if profile and command == "train-arm" else []
    print(f"{command} {out.name}", flush=True)
    with patch("sys.argv", [SUITE, command, "--out", str(out), *batch_options, *profile_options]):
        main()
    result = json.loads(out.read_text())
    if "profiler" in result:
        annotate_payload(result["profiler"], batch)
        out.write_text(json.dumps(result, indent=2) + "\n")
    summary = {key: result[key] for key in ("status", "median_ms", "error") if key in result}
    print(summary, flush=True)
    return checked(result, command)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=(128, 256, 512, 1024))
    parser.add_argument("--arm", choices=ARMS, nargs="+", default=("baseline", "compute", "cached"))
    parser.add_argument("--forward")
    parser.add_argument("--backward")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    os.environ.pop("LNET_VSCAN_DECOUPLED", None)
    arms = {name: ARMS[name] for name in args.arm}
    if args.forward and args.backward:
        arms["selected"] = (args.forward, args.backward)
    bench = runpy.run_path(SUITE)["main"]
    if args.graph:
        import va_model_builder as builder  # noqa: PLC0415

        build_optimizer = builder.build_optimizer

        def capture_optimizer(*args: object, **kwargs: object) -> torch.optim.Optimizer:
            optimizer = build_optimizer(*args, **kwargs)
            for group in optimizer.param_groups:
                group["capturable"] = True
            return optimizer

        builder.build_optimizer = capture_optimizer
        bench.__globals__["measure"] = graph_measure

    args.out.mkdir(exist_ok=True, parents=True)
    probe = bench.__globals__["probe"](SimpleNamespace())
    (args.out / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    outcome = []
    if not args.verify_only:
        for index, batch in enumerate(args.batch):
            names = list(arms)
            for name in names if index % 2 == 0 else reversed(names):
                output = args.out / "arms" / f"b{batch}_{name}.json"
                outcome.append(run(bench, arms[name], "train-arm", output, batch, profile=True))
    if not args.train_only:
        for name, arm in arms.items():
            for command in ("dynamo", "infer-fold"):
                output = args.out / "infer" / f"{command}_{name}.json"
                outcome.append(run(bench, arm, command, output, min(args.batch), profile=False))
    raise SystemExit(0 if all(outcome) else 1)


if __name__ == "__main__":
    main()
