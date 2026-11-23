"""Interleave identically initialized Wave 14A and Wave 16C training steps."""

import argparse
import gc
import importlib.util
import json
import math
import runpy
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

from lnet.memory_backbone import ChannelRMSNorm, SwiGLU

SUITE = runpy.run_path("h200/liverms_v2_opt/bench_h200.py")
GIB = 1024**3


def baseline_classes(path: Path) -> dict:
    name = "lnet._wave14a_memory_backbone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return {cls: getattr(module, cls.__name__) for cls in (ChannelRMSNorm, SwiGLU)}


def training_case(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0915
    import va_model_builder as builder  # noqa: PLC0415

    size = (4, 512, 768) if args.dense else (args.batch, 224, 224)
    batch, height, width = size
    torch.manual_seed(501)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    image = torch.randn(batch, 3, height, width, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    labels = torch.arange(batch, device="cuda") % 100
    base_types = baseline_classes(args.baseline)
    models = {}
    steps = {}
    optimizers = {}
    for arm in ("baseline", "candidate"):
        torch.manual_seed(937)
        model = SUITE["build_model"](args.variant)
        replacements = base_types if arm == "baseline" else {
            cls: baseline for cls, baseline in base_types.items()
            if cls.__name__ in args.revert_candidate
        }
        for layer in model.modules():
            if type(layer) in replacements:
                layer.__class__ = replacements[type(layer)]
        models[arm] = model
        compiled = torch.compile(model, fullgraph=True)
        opt = builder.build_optimizer(model, {
            "learning_rate": 3e-3,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "fused_optimizer": True,
        })
        steps[arm] = compiled
        optimizers[arm] = opt

    def step(arm: str, *, update: bool = True) -> torch.Tensor:
        model, opt = models[arm], optimizers[arm]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(steps[arm](image)[0], labels)
        loss.backward()
        if update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss.detach()

    arms = ("baseline", "candidate")
    for arm in arms:
        for _ in range(3):
            step(arm)
    torch.cuda.synchronize()
    if args.profile:
        # Initialize CUPTI before capturing either arm.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
            for arm in arms:
                step(arm)
            torch.cuda.synchronize()
    per_arm = {arm: {"ms": [], "allocated_gib": [], "loss": []} for arm in arms}
    for i in range(args.repeats):
        for arm in arms[::1 if i % 2 == 0 else -1]:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            start = time.perf_counter()
            loss = step(arm)
            torch.cuda.synchronize()
            row = per_arm[arm]
            row["ms"].append((time.perf_counter() - start) * 1000)
            row["allocated_gib"].append((torch.cuda.max_memory_allocated() - before) / GIB)
            row["loss"].append(float(loss))

    # Remove the inactive model and optimizer storage from the observed peak.
    for arm in arms:
        other = next(name for name in arms if name != arm)
        resident = [
            *models[other].parameters(), *models[other].buffers(),
            *(value for state in optimizers[other].state.values() for value in state.values()
              if isinstance(value, torch.Tensor)),
        ]
        unique_storage = {
            item.untyped_storage().data_ptr(): item.untyped_storage().nbytes()
            for item in resident if item.is_cuda
        }
        torch.cuda.reset_peak_memory_stats()
        step(arm)
        peak = torch.cuda.max_memory_allocated()
        step(arm, update=False)
        grads = [item.grad for item in models[arm].parameters() if item.grad is not None]
        row = per_arm[arm]
        row.update(
            median_ms=statistics.median(row["ms"]),
            peak_allocated_gib=(peak - sum(unique_storage.values())) / GIB,
            finite_losses=all(math.isfinite(loss) for loss in row["loss"]),
            finite_gradients=bool(grads) and all(bool(item.isfinite().all()) for item in grads),
        )
        optimizers[arm].zero_grad(set_to_none=True)
        del grads
        gc.collect()
    if args.profile:
        add_profiles(per_arm, step)
    return {
        "shape": size, "variant": args.variant,
        "candidate_reverts": args.revert_candidate, "arms": per_arm,
    }


def add_profiles(per_arm: dict, step: Callable) -> None:
    arms = tuple(per_arm)
    profiles = {arm: [] for arm in arms}
    for i in range(3):
        for arm in arms[::1 if i % 2 == 0 else -1]:
            profiles[arm].append(SUITE["profile_step"](partial(step, arm)))
    for arm, samples in profiles.items():
        ordered = sorted(samples, key=lambda data: sum(data["self_cuda_ms_by_bucket"].values()))
        per_arm[arm]["profile_samples"] = samples
        per_arm[arm]["profiler"] = ordered[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=SUITE["VARIANTS"], default=SUITE["VARIANTS"][0])
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--revert-candidate", nargs="*", choices=("SwiGLU", "ChannelRMSNorm"), default=(),
    )
    args = parser.parse_args()
    result = training_case(args)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
