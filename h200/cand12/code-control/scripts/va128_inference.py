"""Prepare fixed-shape VA inference with explicit or measured execution plans."""
from __future__ import annotations

# Canonical builders and private compiler configuration are repository-local contracts.
# pyright: reportImplicitRelativeImport=false
import copy
import math
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
from va128_packed_backbone import PackedVA128Backbone
from va_inference_identity import default_cache_dir, inference_signature
from va_inference_plan import ExecutionPlan, plan_id, read_plan_cache, signature_sha, validate_plan
from va_inference_unified import apply_execution_plan, collect_catalog, structural_catalog

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_inference_spatial import factorized_spatial_inference
from lnet.pac_kernel_launch_config import LaunchGeometry, temporary_launch_geometries

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

FAMILY_PARAMETERS = {32: 573_192, 48: 1_004_744, 64: 1_562_760, 96: 3_253_224, 128: 5_083_176}
ReaderScanFusion = Literal["auto", "none", "rms_scan", "spatial_scan", "scaled_scan"]


class _InferenceReader(FactorizedComplexConv2dReader):
    def _factorized_triton_conv_packed(self, packed_input: Tensor) -> Tensor:
        projected = torch.nn.functional.linear(packed_input, self._pointwise_kernel()).movedim(-1, 1)
        real, imag = self._normalized_spatial_factors(projected.dtype)
        return factorized_spatial_inference(projected, real, imag)


def make_inference_snapshot(model: nn.Module, device: torch.device) -> nn.Module:
    """Copy and prepare constants without changing the trainable source model."""
    frozen = copy.deepcopy(model).eval()
    for module in frozen.modules():
        if getattr(module, "_lnet_untraced_parametrization", False):
            delattr(module, "forward")
            delattr(module, "_lnet_untraced_parametrization")
    frozen.to(device, memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        cast("PackedVA128Backbone", frozen.model).prepare_for_inference_()
    return frozen


def _select_plan(
    signature: dict[str, object], catalog: list[dict[str, object]], cache_dir: str | Path,
) -> tuple[ExecutionPlan, str, str]:
    cached = read_plan_cache(cache_dir, signature, catalog=catalog)
    if cached.status == "invalid":
        raise ValueError(f"invalid inference tuning cache: {cached.reason}")
    if cached.status == "hit":
        if cached.plan is None:
            raise ValueError("invalid inference tuning cache: a hit has no execution plan")
        measured = cached.measurements
        if not isinstance(measured, dict):
            raise ValueError("invalid inference tuning measurements: expected an object")
        if measured.get("status") != "ok" or measured.get("finite") is not True:
            raise ValueError("invalid inference tuning measurements: evaluation did not succeed")
        if measured.get("plan_id") != plan_id(cached.plan):
            raise ValueError("invalid inference tuning measurements: plan identity does not match")
        try:
            matches = signature_sha(measured.get("signature")) == signature_sha(signature)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid inference tuning measurements: malformed signature") from error
        if not matches:
            raise ValueError("invalid inference tuning measurements: signature does not match")

        def finite_number(value: object) -> bool:
            if type(value) not in (int, float):
                return False
            try:
                return math.isfinite(cast("int | float", value))
            except OverflowError:
                return False

        median = measured.get("median_ms")
        samples = measured.get("samples_ms")
        if (
            not finite_number(median) or cast("float", median) <= 0
            or not isinstance(samples, list) or not samples
            or any(not finite_number(sample) or sample <= 0 for sample in samples)
        ):
            raise ValueError("invalid inference tuning measurements: timings must be finite and positive")
        relative = measured.get("relative_l2")
        limit = measured.get("relative_l2_limit")
        if (
            not finite_number(relative) or not finite_number(limit)
            or not 0 <= cast("float", relative) <= cast("float", limit) <= 1
        ):
            raise ValueError("invalid inference tuning measurements: numerical tolerance failed")
        return cached.plan, "cache", cached.reason
    # A missing measurement is explicit, not a guessed width/batch fastpath.
    return ExecutionPlan(), cached.status, cached.reason


class VAInference:
    """Own a fixed-shape inference snapshot, never a training/checkpoint interface.

    An explicit plan is replayed exactly. Otherwise automatic mode reads an
    identity-matched measured plan; a cache miss uses the unmodified baseline
    and exposes that status. Run the joint tuner to populate a missing record.
    """

    def __init__(
        self, model: nn.Module, example: Tensor, *,
        tuning: Literal["auto", "standard", "k128"] = "auto",
        reader_scan_fusion: ReaderScanFusion = "auto",
        plan: ExecutionPlan | None = None, cache_dir: str | Path | None = None,
    ) -> None:
        if (not example.is_cuda or example.dtype != torch.float32 or example.ndim != 4
                or example.shape[0] < 1 or tuple(example.shape[1:]) != (3, 224, 224)):
            raise ValueError("VA inference requires CUDA FP32 [batch, 3, 224, 224] inputs")
        if not isinstance(getattr(model, "model", None), PackedVA128Backbone):
            raise TypeError("prepare the canonical packed VA-family logits adapter")
        backbone = cast("PackedVA128Backbone", model.model)
        self.width = cast("SameResolutionPoleScanBlock", backbone.same_resolution_blocks["56"]).modes
        if sum(parameter.numel() for parameter in model.parameters()) != FAMILY_PARAMETERS.get(self.width):
            raise ValueError("inference optimization requires an original VA family topology")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise TypeError("load the original FP32 checkpoint before preparing BF16 inference")
        if tuning not in ("auto", "standard", "k128"):
            raise ValueError("unknown inference tuning profile")
        if reader_scan_fusion not in ("auto", "none", "rms_scan", "spatial_scan", "scaled_scan"):
            raise ValueError("unknown Reader/scan fusion")
        if plan is not None and (tuning != "auto" or reader_scan_fusion != "auto"):
            raise ValueError("an explicit execution plan cannot be combined with legacy overrides")
        self.shape, self.device = tuple(example.shape), example.device
        self.signature: dict[str, object] | None = None
        catalog = structural_catalog(model)
        if plan is not None:
            self.tuning_source, self.tuning_reason = "explicit", "caller supplied the complete plan"
        elif tuning != "auto" or reader_scan_fusion != "auto":
            plan = ExecutionPlan(
                reader="none" if reader_scan_fusion == "auto" else reader_scan_fusion,
                launch="wide" if tuning == "k128" else "standard",
            )
            self.tuning_source, self.tuning_reason = "explicit", "legacy overrides converted to a plan"
        else:
            self.signature = inference_signature(model, example)
            plan, self.tuning_source, self.tuning_reason = _select_plan(
                self.signature, catalog, cache_dir or default_cache_dir(),
            )
        validate_plan(plan, catalog)
        self.plan = plan
        self.reader_scan_fusion = plan.reader
        frozen = make_inference_snapshot(model, self.device)
        if any(block.scan != "split" for block in plan.blocks):
            catalog = collect_catalog(frozen, example)
        self.applied_fusions = apply_execution_plan(frozen, plan, catalog)
        # These are selectable launch alternatives, not model/batch dispatch rules.
        if plan.launch in ("spatial", "wide"):
            for module in frozen.modules():
                if type(module) is FactorizedComplexConv2dReader:
                    module.__class__ = _InferenceReader
        geometries = {}
        if plan.launch in ("scan", "wide"):
            geometries = {
                "bidirectional_product_scan_forward": LaunchGeometry.build(
                    num_warps=4, blocks={"BLOCK_MODES": 32},
                ),
                "product_scan_path_collapse_forward": LaunchGeometry.build(
                    num_warps=2, blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 4},
                ),
            }
        self._compiled = torch.compile(
            frozen, dynamic=False, fullgraph=True,
            options={"freezing": True, "triton.cudagraphs": True, "fx_graph_cache": False},
        )
        # Keep context-dependent geometry out of stale FX/AOT graph reuse.
        with (inductor_config.patch(freezing=True),
              functorch_config.patch(enable_autograd_cache=False),
              temporary_launch_geometries(geometries)):
            for _ in range(3):
                self(example)
        torch.cuda.synchronize(self.device)

    def __call__(self, images: Tensor) -> Tensor:
        if images.device != self.device or images.dtype != torch.float32:
            raise ValueError("input device and FP32 dtype must match the prepared inference snapshot")
        if tuple(images.shape) != self.shape:
            raise ValueError("input shape changed; prepare another inference snapshot")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.compiler.cudagraph_mark_step_begin()
            return self._compiled(images.contiguous(memory_format=torch.channels_last))


VA128Inference = VAInference


def prepare_va_inference(
    model: nn.Module, example: Tensor, *,
    tuning: Literal["auto", "standard", "k128"] = "auto",
    reader_scan_fusion: ReaderScanFusion = "auto",
    plan: ExecutionPlan | None = None, cache_dir: str | Path | None = None,
) -> VAInference:
    return VAInference(model, example, tuning=tuning, reader_scan_fusion=reader_scan_fusion,
                       plan=plan, cache_dir=cache_dir)


def prepare_va128_inference(model: nn.Module, example: Tensor) -> VAInference:
    return VAInference(model, example)
