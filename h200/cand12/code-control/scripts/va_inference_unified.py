"""Structural catalog and exact site application for measured inference plans."""

from __future__ import annotations

# Installers deliberately share their legacy private snapshot classes.
# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from va_inference_memory_projection import use_inference_projection
from va_inference_plan import BlockPlan, ExecutionPlan, validate_plan
from va_inference_reader_scan import Fusion, enable_reader_scan_fusion
from va_inference_scan2d import enable_scan2d_fusion
from va_inference_vertical_projection import enable_vertical_projection

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_complex_layers import ComplexLinear
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import (
    GatedComplexPostFusion,
    GatedPoleExcitationS2DTransition,
    GatedStrictComplexPostFusion,
)
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_postfusion_inference import (
    FrozenGatedPostFusion,
    PostFusionVariant,
    enable_postfusion_inference,
)
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from lnet.pac_triton_memory_projection_inference import ProjectionFusion


def _catalog_node(site: str, module: nn.Module) -> dict[str, object]:
    if not site:
        raise ValueError("catalog scan nodes must be named children of a snapshot")
    if type(module) is SameResolutionPoleScanBlock:
        block = module
        reader = block.reader
        scan = block.pole_scan
        collapse = block.path_collapse
        projection = block.memory_projection
        post = block.post_fusion
        post_site = f"{site}.post_fusion"
        kind = "same_resolution"
        modes, poles = block.modes, block.pole_modes
    elif type(module) is ComplexScanStage:
        scan = module
        reader = scan.pole_input_projection
        poles = scan.modes
        if scan.output_modes is None:
            kind = "terminal"
            modes = getattr(reader, "input_modes", None)
            collapse, projection, post, post_site = None, None, None, None
        else:
            transition = scan.augmented
            if type(transition) is not GatedPoleExcitationS2DTransition:
                raise TypeError(f"unsupported transition type at {site}")
            kind = "transition"
            modes = transition.excitation_modes
            collapse = scan.quadrant_path_mode_combiner
            projection = transition.memory_projection
            post = transition.post_fusion
            post_site = f"{site}.augmented.post_fusion"
    else:
        raise TypeError(f"unsupported scan node type at {site}: {type(module).__name__}")
    if type(reader) is not FactorizedComplexConv2dReader:
        raise TypeError(f"unsupported Reader type at {site}")
    if type(scan) is not ComplexScanStage:
        raise TypeError(f"unsupported pole scan type at {site}")
    if collapse is not None and type(collapse) is not PathOnlyCollapse:
        raise TypeError(f"unsupported path collapse type at {site}")
    if projection is not None and type(projection) is not ComplexLinear:
        raise TypeError(f"unsupported memory projection type at {site}")
    if post is not None and type(post) not in (
        GatedComplexPostFusion, GatedStrictComplexPostFusion,
    ):
        raise TypeError(f"unsupported PostFusion type at {site}")
    if scan.readout_mode not in ("directional", "mean_energy"):
        raise ValueError(f"unsupported packed readout mode at {site}")
    return {
        "site": site,
        "kind": kind,
        "memory_projection": projection is not None,
        "post_site": post_site,
        "modes": modes,
        "poles": poles,
        "hidden": None if post is None else post.hidden_modes,
        "readout_mode": scan.readout_mode,
        "gain_normalization": scan.product_gain_normalization,
        "path_swiglu": bool(getattr(collapse, "path_swiglu", False)),
        "path_collapse": collapse is not None,
        "reader_rank": reader.rank,
        "reader_kernel_size": reader.kernel_size,
        "reader_match_input_rms": reader.match_input_rms,
    }


def structural_catalog(model: nn.Module) -> list[dict[str, object]]:
    """Describe original packed scan boundaries without executing the model.

    Descendant pole coefficient builders are not independent execution sites.
    Exact type checks reject customized scan equations rather than silently
    substituting a superficially compatible forward implementation.
    """
    result = []
    boundaries: list[str] = []
    for site, module in model.named_modules():
        if any(site.startswith(f"{parent}.") for parent in boundaries):
            continue
        if isinstance(module, SameResolutionPoleScanBlock) or (
            isinstance(module, ComplexScanStage) and module.pole_input_projection is not None
        ):
            result.append(_catalog_node(site, module))
            boundaries.append(site)
    return result


def _tensor_metadata(value: object, *, dtype: bool) -> object:
    if isinstance(value, Tensor):
        return str(value.dtype) if dtype else list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(item, dtype=dtype) for item in value]
    if value is None:
        return None
    raise TypeError("catalog forward hooks require tensors, sequences, or None")


def collect_catalog(snapshot: nn.Module, example: Tensor) -> list[dict[str, object]]:
    """Record one inference forward, keeping only JSON shape/dtype metadata.

    CUDA uses BF16 autocast. CPU supports lightweight structural test fixtures.
    Hook handles are removed even when the forward raises.
    """
    if snapshot.training:
        raise ValueError("catalog collection requires an evaluation snapshot")
    catalog = structural_catalog(snapshot)
    handles = []
    try:
        for node in catalog:
            site = cast("str", node["site"])

            def capture(
                _module: nn.Module, inputs: tuple[object, ...], output: object,
                *, record: dict[str, object] = node,
            ) -> None:
                if "input_shape" in record:
                    raise ValueError(f"scan node executed multiple times: {record['site']}")
                record["input_shape"] = _tensor_metadata(inputs[0], dtype=False)
                record["input_dtype"] = _tensor_metadata(inputs[0], dtype=True)
                record["output_shape"] = _tensor_metadata(output, dtype=False)
                record["output_dtype"] = _tensor_metadata(output, dtype=True)

            handles.append(snapshot.get_submodule(site).register_forward_hook(capture))
        with torch.inference_mode(), torch.autocast(example.device.type, dtype=torch.bfloat16):
            snapshot(example)
    finally:
        for handle in handles:
            handle.remove()
    if any("input_shape" not in node for node in catalog):
        raise ValueError("catalog contains scan nodes not exercised by the example")
    return catalog


class _SingleNode(nn.Module):
    """Installer view exposing one execution boundary and no descendants."""

    def __init__(self, node: nn.Module) -> None:
        super().__init__()
        self.node = node

    def modules(self) -> Iterator[nn.Module]:
        yield self.node


def _spatial_shape(node: dict[str, object]) -> tuple[int, int]:
    shape = node.get("input_shape")
    if (
        not isinstance(shape, list) or len(shape) != 4
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError(f"selected scan fusion requires observed NHWC input_shape: {node['site']}")
    return cast("int", shape[1]), cast("int", shape[2])


def _validate_snapshot(
    snapshot: nn.Module, plan: ExecutionPlan, catalog: Sequence[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, FrozenGatedPostFusion]]:
    if snapshot.training:
        raise ValueError("execution plans require an isolated evaluation snapshot")
    validate_plan(plan, catalog)
    fresh = structural_catalog(snapshot)
    supplied = {cast("str", node["site"]): node for node in catalog}
    if supplied.keys() != {node["site"] for node in fresh}:
        raise ValueError("catalog sites do not match the snapshot")
    for node in fresh:
        recorded = supplied[cast("str", node["site"])]
        if any(recorded.get(key) != value for key, value in node.items()):
            raise ValueError(f"catalog structure does not match the snapshot: {node['site']}")
    for block in plan.blocks:
        node = supplied[block.site]
        if block.scan == "scan2d" and max(_spatial_shape(node)) > 56:
            raise ValueError(f"scan2d spatial dimensions exceed the kernel cap: {block.site}")
        if block.scan == "vertical_projection":
            height, _ = _spatial_shape(node)
            if height > 56:
                raise ValueError(f"unsupported vertical projection geometry: {block.site}")
    # Build every replacement before mutating the snapshot. This validates
    # PostFusion classes, hooks, evaluation state, and frozen-weight construction.
    replacements = {}
    for post in plan.posts:
        if post.variant != "none":
            source = snapshot.get_submodule(post.site)
            if source.training:
                raise ValueError(f"selected PostFusion requires evaluation mode: {post.site}")
            staging = nn.ModuleDict({"selected": source})
            staging.training = False
            enable_postfusion_inference(
                staging, cast("PostFusionVariant", post.variant), sites={"selected"},
            )
            replacements[post.site] = cast("FrozenGatedPostFusion", staging["selected"])
    return supplied, replacements


def apply_execution_plan(
    snapshot: nn.Module, plan: ExecutionPlan, catalog: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Validate all choices, then install only their exact structural sites.

    The caller supplies a private, original-class inference snapshot. No
    width/batch policy is embedded here. Launch geometry is handled by the
    enclosing inference API; this function reports its requested profile.
    """
    nodes, replacements = _validate_snapshot(snapshot, plan, catalog)
    choices = {block.site: block for block in plan.blocks}
    audit: list[dict[str, object]] = []
    for site, node in nodes.items():
        block = choices.get(site, BlockPlan(site))
        reader = plan.reader if block.reader == "inherit" else block.reader
        selected = _SingleNode(snapshot.get_submodule(site))
        if reader != "none":
            enable_reader_scan_fusion(selected, cast("Fusion", reader))
        if block.scan == "scan2d":
            enable_scan2d_fusion(
                selected, block_modes=block.scan_block_modes, num_warps=block.scan_num_warps,
                max_size=max(_spatial_shape(node)),
            )
        elif block.scan == "vertical_projection":
            enable_vertical_projection(
                selected, output_tile=block.vertical_output_tile,
                pole_tile=block.vertical_pole_tile, num_warps=block.vertical_num_warps,
                max_height=_spatial_shape(node)[0],
            )
        if block.projection != "none":
            use_inference_projection(
                selected, cast("ProjectionFusion", block.projection),
                block_rows=block.projection_block_rows, output_tile=block.projection_output_tile,
                num_warps=block.projection_num_warps,
            )
        audit.append({
            "site": site, "reader": reader, "scan": block.scan,
            "projection": block.projection, "launch": plan.launch,
        })
    for post in plan.posts:
        if post.site in replacements:
            snapshot.set_submodule(post.site, replacements[post.site])
        audit.append({"site": post.site, "post": post.variant})
    return audit


__all__ = ["apply_execution_plan", "collect_catalog", "structural_catalog"]
