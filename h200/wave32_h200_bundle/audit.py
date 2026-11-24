"""Run the pinned benchmark with Wave 32 profiler classification."""

import argparse
import os
import re
import runpy
from collections import Counter
from collections.abc import Callable

SCAN_NAME = re.compile(r"vscan|project_path|path_mlp", re.IGNORECASE)
READER_NAME = re.compile(r"reader_|factorized_reader|dweight", re.IGNORECASE)


def classify(name: str) -> tuple[str | None, str | None]:
    name = name.lower()
    if SCAN_NAME.search(name):
        if "vscan_tc_reduce" in name:
            return "general_vscan", "tc_reduce"
        group = next(
            (group for group in (
                "saved_fused_tc", "fused_tc", "project_path_tc", "pure_vscan",
                "path_mlp", "project_path", "saved_fused",
            ) if group in name),
            "legacy_vscan",
        )
        phase = next(
            (label for token, label in (
                ("reduce", "reduce"), ("backward", "bwd"), ("forward", "fwd"),
            ) if token in name),
            "other",
        )
        return "general_vscan", f"{group}_{phase}"
    if READER_NAME.search(name):
        group = next(
            (op for op in ("gram_normalize", "reduce", "dinput", "dweight", "forward")
             if op in name),
            "other",
        )
        return "factorized_reader", f"reader_{group}"
    if name.startswith("conv_stem_"):
        return "fused_stem", None
    if name.startswith(("_rmsnorm_", "_swiglu_vec2d_")):
        return "triton", None
    return None, None


def main() -> None:
    namespace = runpy.run_path(os.environ["VA_T93_BENCH"])["main"].__globals__
    kernel_bucket = namespace["kernel_bucket"]
    profile_step = namespace["profile_step"]
    train_arm = namespace["train_arm"]

    def bucket(name: str) -> str:
        return classify(name)[0] or kernel_bucket(name)

    def profile(step: Callable, **kwargs: bool) -> dict:
        result = profile_step(step, **kwargs)
        groups = Counter()
        cuda = result["self_cuda_ms_by_bucket"]
        dispatches = result["dispatches_by_profiler_bucket"]
        for row in result.pop("kernels"):
            name = row["name"]
            elapsed = row["self_cuda_ms"]
            if name in {"va_t93_step", "va_t101_backward"} or name.startswith(
                ("## Call CompiledFxGraph ", "Optimizer.step#"),
            ):
                cuda[bucket(name)] -= elapsed
                dispatches[bucket(name)] -= row["count"]
                continue
            _, subgroup = classify(name)
            if subgroup:
                groups[subgroup] += elapsed
        result.pop("top_kernels", None)
        groups["legacy_vscan_scan_estimated"] = max(
            0.0,
            cuda["general_vscan"] - sum(
                elapsed for group, elapsed in groups.items() if not group.startswith("reader_")
            ),
        )
        result["major_kernel_ms"] = dict(groups)
        return result

    def train(args: argparse.Namespace) -> dict:
        row = train_arm(args)
        if row.get("profiler"):
            cuda = row["profiler"]["self_cuda_ms_by_bucket"]
            row["profiler"]["wall_minus_cuda_ms"] = row["median_ms"] - sum(cuda.values())
        return row

    namespace.update(kernel_bucket=bucket, profile_step=profile, train_arm=train)
    namespace["main"]()


if __name__ == "__main__":
    main()
