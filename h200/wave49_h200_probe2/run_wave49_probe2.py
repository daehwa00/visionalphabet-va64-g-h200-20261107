"""Whole H200 Wave 49 full-model, Reader convolution and vertical scan probes."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter

from typing_extensions import override

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / part) for part in ("src", "scripts", "h200/wave49_h200_screen")]
if not hasattr(typing, "override"):
    typing.override = override

from run_wave49_h200_screen import (
    GIB, SCAN_ENV, attempt, describe_launch, save, scan_inputs, utc,
    summarize_trace as original_summarize_trace,
)
from run_wave49_h200_screen import profile_step as base_profile_step
import run_wave49_h200_screen as screen
import torch
import torch.nn.functional as F

ARMS = {
    "arm_off": ("off", "off", False),
    "arm_proven_floor": ("off", "proven_floor", False),
    "arm_dense_nhwc": ("dense_nhwc", "proven_floor", False),
}


def summarize_trace(events):
    result = original_summarize_trace(events)
    result["below_0p10_ms_total"] = result.pop("below_0p05_ms_total") + sum(
        item["total_ms"] for item in result["kernels"] if item["total_ms"] < 0.10
    )
    result["kernels"] = [item for item in result["kernels"] if item["total_ms"] >= 0.10]
    return result


def profile_step(torch, scans, step):
    screen.summarize_trace = summarize_trace
    try:
        return base_profile_step(torch, scans, step)
    finally:
        screen.summarize_trace = original_summarize_trace


def fullmodel(torch, output: Path, device="cuda", batch=1024) -> dict:
    from va_model_builder import build_dw_terminal_rmsnorm
    from va_model_builder import build_optimizer
    from lnet.model_w36b import prepare_model_w36b_

    from lnet import general_cuda_scans as scans

    os.environ.update(SCAN_ENV, LNET_FUSED_FACTORIZED_READER="1", LNET_READER_VEC="auto")
    generator = torch.Generator(device=device).manual_seed(169)
    inputs = torch.randn((batch, 3, 224, 224), device=device, generator=generator).contiguous(
        memory_format=torch.channels_last,
    )
    targets = torch.randint(100, (batch,), device=device, generator=generator)
    reference = {}
    report = {"batch": batch, "dtype": "bf16", "arms": {}, "scan_env": SCAN_ENV}

    def arm(name: str) -> dict:
        execution, optimization, benchmark = ARMS[name]
        os.environ.update(LNET_READER_EXEC_MODE=execution, VA_READER_OPT_MODE=optimization)
        torch.backends.cudnn.benchmark = benchmark
        torch.manual_seed(169)
        model = (
            build_dw_terminal_rmsnorm(factorized_reader=True)
            .to(
                device=device,
                memory_format=torch.channels_last,
            )
            .train()
        )
        params = dict(model.named_parameters())
        prepare_model_w36b_(model)
        optimizer = build_optimizer(model, {
            "learning_rate": 3e-3,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "fused_optimizer": True,
        })

        def backward():
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(inputs)[0], targets)
            loss.backward()
            return loss

        def step() -> None:
            loss = backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            return loss.detach()

        if device == "cpu":
            loss = step()
            finite = all(bool(p.isfinite().all()) for p in params.values())
            updated = len(optimizer.state)
            if not (finite and bool(loss.isfinite()) and updated == 193):
                raise RuntimeError(f"CPU smoke failed: {loss=}, {finite=}, {updated=}")
            return {"loss": float(loss), "adamw_updated_parameters": updated}

        # Check gradients before any optimizer updates and reuse this as warmup 1.
        loss = backward().item()
        grads = {
            key: param.grad.detach().float().cpu()
            for key, param in params.items()
            if param.grad is not None
        }
        finite = sum(bool(grad.isfinite().all()) for grad in grads.values())
        sanity = {
            "loss": loss if math.isfinite(loss) else None,
            "grad_count": len(grads),
            "finite_grads": finite,
        }
        if name == "arm_off":
            reference.update(loss=loss, grads=grads)
        elif "grads" in reference and finite == len(grads):
            baseline = reference["grads"]
            if grads.keys() == baseline.keys():
                pairs = [(grad, baseline[key]) for key, grad in grads.items()]
                difference = sum(float((grad - ref).square().sum()) for grad, ref in pairs)
                scale = sum(float(ref.square().sum()) for _, ref in pairs)
                rel_l2 = math.sqrt(difference / max(1e-30, scale))
                loss_error = abs(loss - reference["loss"])
                if math.isfinite(rel_l2) and math.isfinite(loss_error):
                    sanity.update(
                        loss_abs_error=loss_error,
                        gradient_rel_l2=rel_l2,
                        gradient_max_parameter_rel_l2=max(
                            float((grad - ref).norm()) / max(1e-30, float(ref.norm()))
                            for grad, ref in pairs
                        ),
                        gradient_max_abs=max(
                            float((grad - ref).abs().max()) for grad, ref in pairs
                        ),
                        parity_pass=finite == 193 and loss_error <= 0.02 and rel_l2 <= 0.04,
                    )
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(4):
            start = perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((perf_counter() - start) * 1_000)
        peak = torch.cuda.max_memory_allocated() / GIB
        profile = (attempt(lambda: profile_step(torch, scans, step))
                   if name != "arm_proven_floor" else None)
        return {
            "step_ms": statistics.median(times),
            "samples_ms": times,
            "peak_allocated_gib": peak,
            "sanity": sanity,
            "profile": profile,
        }

    if device == "cpu":
        return arm("arm_off")
    for name in ARMS:
        report["arms"][name] = attempt(partial(arm, name))
        torch.cuda.empty_cache()
        save(output / "fullmodel_unfiltered.json", report)
    return report



def graph_time(run, flush):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(8):
            flush.add_(1)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def comparison(actual, expected, atol=0.02, rtol=0.02):
    # Limit FP32 diagnostic workspace even for the full-resolution activations.
    square_error = square_reference = max_abs = 0.0
    exact = close = True
    for value, ref in zip(actual, expected):
        for x, y in zip(value.split(8), ref.split(8)):
            exact &= torch.equal(x, y)
            a, b = x.float(), y.float()
            diff = (a - b).abs()
            error, scale, maximum, passes = torch.stack((
                diff.square().sum(), b.square().sum(), diff.max(),
                (diff <= atol + b.abs() * rtol).all().float(),
            )).tolist()
            square_error += error
            square_reference += scale
            max_abs = max(max_abs, maximum)
            close &= bool(passes)
    relative = math.sqrt(square_error / max(1e-30, square_reference))
    return {"exact": exact, "pass": close, "max_abs": max_abs,
            "relative_l2": relative if math.isfinite(relative) else None}


def im2col(x, kernel):
    b, c, h, w = x.shape
    padded = F.pad(x.movedim(1, -1), (0, 0, 1, 1, 1, 1))
    patches = torch.stack(
        [padded[:, row:row+h, col:col+w, :] for row in range(3) for col in range(3)],
        dim=-2,
    ).reshape(-1, 9 * c)
    matrix = kernel.permute(0, 2, 3, 1).reshape(kernel.shape[0], -1).t()
    return (patches @ matrix).reshape(b, h, w, -1).movedim(-1, 1)


def symbols(run):
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profile:
        run()
        torch.cuda.synchronize()
    return sorted({event.name for event in profile.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA})


def timed_autograd(forward, variables, gradient, flush):
    y = forward()
    def backward():
        return torch.autograd.grad(y, variables, gradient, retain_graph=True)
    def step():
        return torch.autograd.grad(forward(), variables, gradient)
    return {"fwd": graph_time(forward, flush), "bwd": graph_time(backward, flush),
            "total": graph_time(step, flush)}


def conv_shape(batch, side, channels, outputs, flush, only_inductor=False):
    from lnet.factorized_complex_scan_reader import FactorizedComplexConv2dReader
    from lnet.memory_backbone import FactorizedMemoryReader
    from va_model_builder import READER_RANK

    torch.manual_seed(172)
    dtype = torch.bfloat16
    fmt = torch.channels_last
    reader = FactorizedMemoryReader(FactorizedComplexConv2dReader(
        channels // 2, outputs // 2, rank=READER_RANK, normalize_input=True,
    )).cuda()
    x = torch.randn((batch, side, side, channels), device="cuda", dtype=dtype)
    x = x.mul_(0.125).movedim(-1, 1).requires_grad_()
    def synthesize():
        return reader._normalized_kernel().to(dtype=dtype, memory_format=fmt)
    k = synthesize().detach().requires_grad_()
    grad = torch.randn((batch, side, side, outputs), device="cuda", dtype=dtype)
    grad = grad.movedim(-1, 1).mul_(0.125)
    def bare(a, b):
        return F.conv2d(a, b, padding=1)
    torch.backends.cudnn.benchmark = False
    expected_y = bare(x, k)
    expected_grads = torch.autograd.grad(expected_y, (x, k), grad)
    expected_y = expected_y.detach()
    flops = 2 * batch * side**2 * channels * outputs * 9

    def candidate(function, benchmark=False, trace=False, planes=False):
        torch.backends.cudnn.benchmark = benchmark
        def forward():
            return function(x, k)
        compare_output = (lambda y: tuple(y.split(160, dim=1))) if planes else lambda y: (y,)
        y = forward()
        vector = tuple(grad.split(160, dim=1)) if planes else grad
        grads = torch.autograd.grad(y, (x, k), vector)
        parity = {
            "output": comparison(y if planes else (y,), compare_output(expected_y)),
            "input_gradient": comparison((grads[0],), (expected_grads[0],)),
            "weight_gradient": comparison((grads[1],), (expected_grads[1],)),
        }
        del y, grads
        timings = timed_autograd(forward, (x, k), vector, flush)
        result = {**timings, "parity": parity, "benchmark": benchmark,
                  "effective_tflops": {
                      name: flops * (3 if name == "total" else 2 if name == "bwd" else 1)
                      / value["median_ms"] / 1e9 for name, value in timings.items()
                  }}
        if trace:
            def masked(dx, dw):
                return torch.ops.aten.convolution_backward.default(
                    grad, x, k, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
                    [dx, dw, False],
                )
            result["symbols"] = {
                "fprop": symbols(forward),
                "backward_combined": symbols(lambda: masked(True, True)),
                "dgrad": symbols(lambda: masked(True, False)),
                "wgrad": symbols(lambda: masked(False, True)),
            }
        return result

    if only_inductor:
        compiled = torch.compile(bare, mode="max-autotune-no-cudagraphs")
        return candidate(compiled, trace=False) | {"fprop_symbols": symbols(lambda: compiled(x, k))}

    def input_norm():
        return reader.input_norm(x.movedim(1, -1))
    results = {
        "input_norm_only": attempt(lambda: timed_autograd(
            input_norm, (x, reader.input_norm.weight), torch.ones_like(x.movedim(1, -1)), flush,
        )),
        "kernel_synth_only": attempt(lambda: timed_autograd(
            synthesize, tuple(p for name, p in reader.named_parameters() if "input_norm" not in name),
            torch.ones_like(k), flush,
        )),
    }
    reference = f"conv2d_o{outputs}_nobench"
    candidates = {
        reference: partial(candidate, bare, trace=True),
        f"conv2d_o{outputs}_bench": partial(candidate, bare, benchmark=True, trace=True),
        "im2col_cublas": partial(candidate, im2col),
    }
    if outputs == 320:
        def padded(a, b):
            return bare(a, F.pad(b, (0, 0, 0, 0, 0, 0, 0, 64)).contiguous(
                memory_format=fmt,
            ))[:, :320]
        candidates.update({
            "pad_o384_nobench": partial(candidate, padded),
            "pad_o384_bench": partial(candidate, padded, benchmark=True),
            "split_256_64": partial(candidate, lambda a, b: torch.cat(
                (bare(a, b[:256]), bare(a, b[256:])), dim=1,
            )),
            "split_160_160": partial(candidate, lambda a, b: (
                bare(a, b[:160]), bare(a, b[160:]),
            ), planes=True),
        })
    for name, run in candidates.items():
        if name.endswith("_bench"):
            # cuDNN v8 keeps its execution-plan cache per thread. Isolate
            # autotuning from the no-benchmark reference for identical shapes.
            torch.cuda.synchronize()
            with ThreadPoolExecutor(max_workers=1) as pool:
                results[name] = pool.submit(attempt, run).result()
        else:
            results[name] = attempt(run)
        torch.cuda.empty_cache()
        if "total" in results.get(reference, {}) and "total" in results[name]:
            results[name]["delta_total_ms"] = (
                results[name]["total"]["median_ms"] - results[reference]["total"]["median_ms"]
            )
    return results


def compile_probe(side, c, o, output):
    target = output / f"inductor_{side}.tmp"
    process = subprocess.run(
        [sys.executable, __file__, "--output", str(output), "--inductor", str(side), str(c), str(o)],
        text=True, capture_output=True, timeout=15, check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-3000:])
    result = json.loads(target.read_text())
    target.unlink()
    return result


def reader_stage(output):
    flush = torch.zeros(128 * 1024**2 // 4, device="cuda", dtype=torch.float32)
    results = {}
    for side, c, o in ((56, 64, 320), (28, 96, 320), (14, 128, 320), (14, 128, 256)):
        key = f"{side}x{side}_{c}_to_{o}"
        results[key] = attempt(partial(conv_shape, 1024, side, c, o, flush))
        if side in {14, 56} and o == 320:
            trial = attempt(partial(compile_probe, side, c, o, output))
            baseline = results[key].get("conv2d_o320_nobench", {})
            if "total" in trial and "total" in baseline:
                trial["delta_total_ms"] = trial["total"]["median_ms"] - baseline["total"]["median_ms"]
            results[key]["inductor_max_autotune"] = trial
        save(output / "reader_conv_supercharger.json", results)
    return results

def scan_stage(output):
    from lnet import general_cuda_scans as scans

    os.environ.update(SCAN_ENV)
    torch.manual_seed(172)
    flush = torch.zeros(128 * 1024**2 // 4, device="cuda", dtype=torch.float32)
    shape, pole, states, gains, path = scan_inputs(torch, 1024, 56, 56)
    device = states.device.index or 0
    max_smem = scans.device_limits(device)[1]
    report = {"shape": shape, "max_dynamic_smem_bytes": max_smem, "cases": {}}
    for coarse in (False, True):
        label = "56c" if coarse else "56d"
        def measure_case():
            side = 28 if coarse else 56
            real, imag = torch.randn((2, shape[0], side, side, shape[-1]),
                                     device="cuda", dtype=states.dtype).unbind(0)
            original = scans._plan_launch
            baseline = original(states, shape, backward=True, vertical=True, coarse=coarse,
                                hidden=8, precise=False)
            if baseline is None:
                raise RuntimeError("No strict_ro launch for this shape")

            def gradients():
                return scans.vertical_backward(pole, states, gains, path, real, imag, coarse)
            reference = gradients()
            def stream_values(values):
                for value in values:
                    if value.ndim == 5:
                        yield from value.unbind(0)
                    else:
                        yield value
            def measure(span=None, warps=None):
                launch = baseline
                if span is not None:
                    key = replace(
                        baseline.key, chunk=span, warps=warps,
                        backward_reduce_warps=min(warps, baseline.key.backward_reduce_warps)
                        if baseline.key.backward_reduce_warps else 0,
                    )
                    if key.backward_reduce_warps == warps:
                        key = replace(key, backward_reduce_warps=0)
                    threads = warps * scans._WARP_SIZE
                    smem = max(threads * 128, scans._sgs_smem(key, shape[1]))
                    if key.sgs:
                        smem += shape[1] * threads * 8
                    if smem > max_smem:
                        return {"skipped": "dynamic shared memory limit", "smem_bytes": smem}
                    launch = baseline._replace(key=key, plan=baseline.plan._replace(
                        chunk=span, warps=warps, smem=smem,
                    ))
                hits = 0
                def force(source, options_shape, **options):
                    nonlocal hits
                    if (source is states and options_shape == shape and options["backward"]
                            and options["vertical"] and options["coarse"] == coarse):
                        hits += 1
                        return launch
                    return original(source, options_shape, **options)
                metadata = describe_launch(torch, scans, launch)
                if not metadata["active_warps_per_sm"]:
                    return {"skipped": "zero resident blocks", "launch": metadata}
                scans._plan_launch = force
                try:
                    actual = gradients()
                    if hits != 1:
                        raise RuntimeError(f"Expected one forced scan launch, observed {hits}")
                    parity = comparison(stream_values(actual), stream_values(reference), 0, 0)
                    del actual
                    timing = graph_time(gradients, flush)
                finally:
                    scans._plan_launch = original
                return {"launch": metadata, "chunk_count": math.ceil(shape[1] / launch.plan.chunk),
                        "parity": parity, "timing": timing}

            results = {"default": attempt(measure)}
            combos = [(28, 4), (28, 2), (56, 4), (56, 2)]
            for span, warps in combos:
                results[f"chunk{span}_warps{warps}"] = attempt(partial(measure, span, warps))
            if results["chunk56_warps4"].get("skipped") == "dynamic shared memory limit":
                results["chunk56_warps1"] = attempt(partial(measure, 56, 1))
            default_timing = results["default"].get("timing", {})
            for value in results.values():
                if "timing" in value and default_timing:
                    value["delta_ms"] = value["timing"]["median_ms"] - default_timing["median_ms"]
            return results
        report["cases"][label] = attempt(measure_case)
        save(output / "vscan_chunk_probe.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--inductor", nargs=3, type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.cpu_smoke:
        report = fullmodel(torch, output, "cpu", 1)
        save(output / "cpu_smoke.json", report)
        print(report)
        return
    props = torch.cuda.get_device_properties(0)
    if "H200" not in props.name or props.total_memory < 120 * GIB:
        raise RuntimeError(f"Expected whole H200, got {props}")
    if args.inductor:
        side, c, o = args.inductor
        flush = torch.zeros(128 * 1024**2 // 4, device="cuda", dtype=torch.float32)
        save(output / f"inductor_{side}.tmp", conv_shape(1024, side, c, o, flush, True))
        return
    meta = {"start_utc": utc(), "gpu": props.name, "memory_bytes": props.total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}
    stages = {
        "fullmodel_unfiltered": partial(fullmodel, torch),
        "reader_conv_supercharger": reader_stage,
        "vscan_chunk_probe": scan_stage,
    }
    for name, run in stages.items():
        print(name, flush=True)
        target = output / f"{name}.json"
        result = attempt(partial(run, output))
        if "error" in result and target.exists():
            result = {**json.loads(target.read_text()), **result}
        save(target, result)
        torch.cuda.empty_cache()
    meta["finish_utc"] = utc()
    save(output / "meta.json", meta)


if __name__ == "__main__":
    main()
