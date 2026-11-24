"""Classify CUDA self time and run the pinned benchmark."""

import argparse
import os
import runpy
from collections import Counter
from collections.abc import Callable

from lnet.model_w35c import apply_inductor_tune

SUBGROUPS = {
    "coop_bwd": "general_vscan",
    "aosoa_pure_fwd": "general_vscan",
    "aosoa_pure_bwd": "general_vscan",
    "aosoa_project_fwd": "general_vscan",
    "aosoa_project_bwd": "general_vscan",
    "hscan_vec2_fwd": "general_hscan",
    "hscan_vec2_bwd": "general_hscan",
    "legacy_vscan_scan_estimated": "general_vscan",
    "legacy_vscan_reduce": "general_vscan",
    "reader_forward": "factorized_reader",
    "reader_dinput": "factorized_reader",
    "reader_dweight": "factorized_reader",
    "reader_reduce": "factorized_reader",
    "reader_gram_normalize": "factorized_reader",
    "fused_stem_stage0_fwd": "fused_stem",
    "fused_stem_stage0_bwd": "fused_stem",
    "fused_stem_stage0_reduce": "fused_stem",
    "fused_stem_stage1": "fused_stem",
    "swiglu_custom": "triton",
    "rmsnorm_bwd_custom": "triton",
    "nr4_correlation_custom": "triton",
    "inductor_triton_other": "triton",
}
PREFIXES = {
    "vscan_coop_bwd": "coop_bwd",
    "vscan_aosoa_pure_fwd": "aosoa_pure_fwd",
    "vscan_aosoa_pure_bwd": "aosoa_pure_bwd",
    "project_path_fwd_aosoa": "aosoa_project_fwd",
    "project_path_bwd_aosoa": "aosoa_project_bwd",
    "hscan_vec2_fwd": "hscan_vec2_fwd",
    "hscan_vec2_bwd": "hscan_vec2_bwd",
    "vscan_legacy": "legacy_vscan_scan_estimated",
    "reduce_vscan_partials": "legacy_vscan_reduce",
    "reader_gram_normalize": "reader_gram_normalize",
    "reader_dweight_reduce": "reader_reduce",
    "reader_forward": "reader_forward",
    "reader_dinput": "reader_dinput",
    "reader_dweight": "reader_dweight",
    "stem_conv_tensor_stage0_fwd": "fused_stem_stage0_fwd",
    "stem_conv_tensor_stage0_bwd": "fused_stem_stage0_bwd",
    "stem_conv_tensor_reduce": "fused_stem_stage0_reduce",
    "conv_stem_forward": "fused_stem_stage0_fwd",
    "conv_stem_backward": "fused_stem_stage0_bwd",
    "conv_stem_reduce": "fused_stem_stage0_reduce",
    "conv_stem_stage1_": "fused_stem_stage1",
    "_swiglu": "swiglu_custom",
    "_packed_swiglu": "swiglu_custom",
    "_rmsnorm_bwd": "rmsnorm_bwd_custom",
    "_channel_rms_grad": "rmsnorm_bwd_custom",
    "_nr4_correlation": "nr4_correlation_custom",
    "triton_": "inductor_triton_other",
}


def classify(name: str) -> tuple[str | None, str | None]:
    # Profiler can demangle templated reader entry points into "void name<...>".
    name = name.removeprefix("void ")
    for prefix, group in PREFIXES.items():
        if name.startswith(prefix):
            return SUBGROUPS[group], group
    if name.startswith("hscan_legacy"):
        return "general_hscan", None
    if name.startswith(("path_mlp_", "reduce_path_mlp_partials")):
        return "general_vscan", None
    if name.startswith("_rmsnorm"):
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
        groups = Counter(dict.fromkeys(SUBGROUPS, 0.0))
        cuda = result["self_cuda_ms_by_bucket"]
        dispatches = result["dispatches_by_profiler_bucket"]
        kernels = []
        for row in result.pop("kernels"):
            name, elapsed = row["name"], row["self_cuda_ms"]
            if name.startswith(("va_t93_step", "va_t101_backward", "## Call CompiledFxGraph ",
                                "Optimizer.step#")):
                cuda[bucket(name)] -= elapsed
                dispatches[bucket(name)] -= row["count"]
                continue
            _, subgroup = classify(name)
            if subgroup:
                groups[subgroup] += elapsed
            kernels.append(row)
        cuda.pop("general_scan_shared", None)
        result["general_cuda_scans_verified"] = bool(result["general_cuda_scans"]) and any(
            row["name"] == "scan" or row["bucket"] in {"general_vscan", "general_hscan"}
            for row in kernels
        )
        result["scan_axis_attribution"] = "Named CUDA kernels; per-shape times are CUDA events"
        result["top_kernels"] = kernels[:15]
        result["major_kernel_ms"] = dict(groups)
        return result

    def train(args: argparse.Namespace) -> dict:
        row = train_arm(args)
        cuda = row["profiler"]["self_cuda_ms_by_bucket"]
        row["profiler"]["wall_minus_cuda_ms"] = row["median_ms"] - sum(cuda.values())
        if os.environ["VA_W35_ARM"] not in {
            "w33_base", "w35c_rmsnorm_bwd_inductor", "w35_best_combo",
            "w35_best_combo_no_decoupled",
        }:
            row["profiler"].pop("top_kernels")
        return row

    namespace.update(kernel_bucket=bucket, profile_step=profile, train_arm=train)
    with apply_inductor_tune():
        namespace["main"]()


if __name__ == "__main__":
    main()
