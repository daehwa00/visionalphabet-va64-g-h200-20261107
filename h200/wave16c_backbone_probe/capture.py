"""Capture all training step dispatches and their host correlations."""

import argparse
import json
import runpy
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import torch
from compare import baseline_classes
from inventory import build_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    suite = runpy.run_path("h200/liverms_v2_opt/bench_h200.py")
    train_arm = suite["train_arm"]
    helpers = train_arm.__globals__
    args.out.mkdir(parents=True, exist_ok=True)

    def capture(step) -> dict:  # noqa: ANN001
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        ) as profiler:
            step()
            torch.cuda.synchronize()
        trace = args.out / "trace.json"
        profiler.export_chrome_trace(str(trace))
        dispatches = build_inventory(json.loads(trace.read_text()))
        (args.out / "inventory.json").write_text(json.dumps(dispatches, indent=2) + "\n")
        by_name = defaultdict(lambda: {"count": 0, "cuda_ms": 0.0})
        for row in dispatches:
            entry = by_name[row["name"], row["bucket"]]
            entry["count"] += 1
            entry["cuda_ms"] += row["duration_us"] / 1000
        rows = [
            {"name": name, "bucket": bucket, **data} for (name, bucket), data in by_name.items()
        ]
        return {"kernels": sorted(rows, key=lambda event: -event["cuda_ms"])}


    def dense_setup(batch: int) -> torch.Tensor:
        torch.manual_seed(501)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        return torch.randn(batch, 3, 512, 768, device="cuda").contiguous(
            memory_format=torch.channels_last
        )

    options = {"profile_step": capture}
    if args.baseline:
        classes = baseline_classes(args.baseline)
        build_model = suite["build_model"]

        def build_baseline(variant: str) -> torch.nn.Module:
            model = build_model(variant)
            for layer in model.modules():
                if type(layer) in classes:
                    layer.__class__ = classes[type(layer)]
            return model

        options["build_model"] = build_baseline
    if args.dense:
        options["setup"] = dense_setup
    task = argparse.Namespace(
        batch=4 if args.dense else 128,
        variant=suite["VARIANTS"][0],
        profile=True,
        out=args.out / "step.json",
    )
    with patch.dict(helpers, options):
        result = train_arm(task)
    task.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
