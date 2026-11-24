"""Run the pinned benchmark with Wave 34 profiler classification."""

import argparse
import os
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

SCAN_NAME = re.compile(r"vscan|project_path|path_mlp")
READER_NAME = re.compile(r"(?:^|\W)reader_")
W34_KERNELS = {
    "modefirst_vscan_forward": ("general_vscan", "w34a_scan_fwd"),
    "modefirst_vscan_backward": ("general_vscan", "w34a_scan_bwd"),
    "_project_forward": ("general_vscan", "w34a_proj_fwd"),
    "_project_backward": ("general_vscan", "w34a_proj_bwd"),
    "_transpose_channels": ("general_vscan", "w34a_transpose"),
    "_reduce_projection": ("general_vscan", "w34a_reduce"),
    "_residual_rmsnorm_backward_kernel": ("triton", "fused_res_norm_bwd"),
    "_rmsnorm_backward_kernel": ("triton", "rmsnorm_bwd_custom"),
    "_swiglu_vec2d_kernel": ("triton", "swiglu_custom"),
    "_nr4_correlation_kernel": ("triton", "nr4_correlation_custom"),
    "_rmsnorm_forward_kernel": ("triton", "rmsnorm_fwd_custom"),
    "_rmsnorm_w34c_forward_kernel": ("triton", "rmsnorm_fwd_custom"),
    "_fused_res_rmsnorm_forward_kernel": ("triton", "fused_res_norm_fwd"),
}


def classify(name: str) -> tuple[str | None, str | None]:
    name = name.lower()
    names = name.partition("(")[0].split()
    symbol = names[-1] if names else ""
    if symbol.startswith("triton_"):
        return "triton", "inductor_triton_other"
    custom = W34_KERNELS.get(symbol) or W34_KERNELS.get(re.sub(r"_\d+$", "", symbol))
    if custom:
        return custom
    if SCAN_NAME.search(name):
        phase = next(
            (label for token, label in (
                ("reduce", "reduce"), ("backward", "bwd"), ("forward", "fwd"),
            ) if token in name),
            "other",
        )
        return "general_vscan", f"legacy_vscan_{phase}"
    if READER_NAME.search(name):
        group = next(
            (op for op in ("gram_normalize", "reduce", "dinput", "dweight", "forward")
             if op in name),
            "other",
        )
        return "factorized_reader", f"reader_{group}"
    if name.startswith("conv_stem_"):
        return "fused_stem", None
    return None, None


def main() -> None:
    benchmark = Path(os.environ["VA_T93_BENCH"])
    source = benchmark.read_text()
    # Only the ambiguous NVRTC "scan" symbol belongs in the event-based split.
    guard = "        if launch is None:\n"
    if source.count(guard) != 1:
        raise RuntimeError("Pinned benchmark dispatch tracker changed")
    source = source.replace(guard, (
        "        if launch is None or scans._kernel_name(launch.key) != 'scan':\n"
    ))
    namespace = {"__file__": str(benchmark), "__name__": "va_t124_benchmark"}
    exec(compile(source, str(benchmark), "exec"), namespace)  # noqa: S102
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
        named_vscan_ms = 0.0
        for row in result.pop("kernels"):
            name = row["name"]
            elapsed = row["self_cuda_ms"]
            if name in {"va_t93_step", "va_t101_backward"} or name.startswith(
                ("## Call CompiledFxGraph ", "Optimizer.step#"),
            ):
                cuda[bucket(name)] -= elapsed
                dispatches[bucket(name)] -= row["count"]
                continue
            category, subgroup = classify(name)
            if category == "general_vscan":
                named_vscan_ms += elapsed
            if subgroup:
                groups[subgroup] += elapsed
        result.pop("top_kernels", None)
        groups["legacy_vscan_scan_estimated"] = max(
            0.0, cuda["general_vscan"] - named_vscan_ms,
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
