"""Full-step validated layouts for the same-resolution 8-channel path adjoint."""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import torch
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .kernel_launch_config import (
    LaunchGeometry,
    geometry_store_signature,
    make_launch_scope,
    validated_geometry,
)
from .triton_product_scan_kernels import (
    _product_scan_coarse4_associative_backward_kernel,
)

LAYOUT_NAME = "product_scan_direct_path_backward_layout"
BUNDLED_LAYOUTS = Path(__file__).with_name("validated_path_layouts.json")
BASELINE = LaunchGeometry.build()


@cache
def _layout_revision() -> str:
    # Both the Triton program and its Python grid/argument mapping affect a saved winner.
    digest = hashlib.sha256(
        _product_scan_coarse4_associative_backward_kernel.cache_key.encode()
    )
    for path in (Path(__file__), Path(__file__).with_name("product_scan_ops.py")):
        # BEGIN R6 DISPATCH
        from .r6_dispatch import legacy_dispatch_source

        digest.update(legacy_dispatch_source(path))
        # R6 LEGACY:         digest.update(path.read_bytes())
        # END R6 DISPATCH
    return digest.hexdigest()


@cache
def _driver_api_version() -> int | None:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        if driver.cuDriverGetVersion(ctypes.byref(version)) == 0:
            return version.value
    except (AttributeError, OSError):
        pass
    return None


def candidate_geometry(
    group: int, modes: int, warps: int, compact: int
) -> LaunchGeometry:
    """Group zero keeps the original grid. Channel bits 1 and 2 enable pack and unpack respectively."""
    if group not in (0, 1, 2, 4, 8, 16, 32, 56) or modes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsupported spatial group or mode tile")
    if warps not in (1, 2, 4, 8) or compact not in (0, 1, 2, 3):
        raise ValueError("Unsupported warp count or path layout")
    return LaunchGeometry.build(
        num_warps=warps,
        blocks={
            "BLOCK_MODES": modes,
            "LINES_PER_GROUP": group,
            "COMPACT_CHANNELS": compact,
        },
    )


def selected_layout(source: torch.Tensor) -> tuple[int, int, LaunchGeometry | None]:
    batch, height, width, modes = source.shape
    compiled = torch.compiler.is_compiling() or get_proxy_mode() is not None
    shape = {
        "batch": batch,
        "height": height,
        "width": width,
        "modes": modes,
        "deterministic": int(torch.are_deterministic_algorithms_enabled()),
        "emulate_precision_casts": int(inductor_config.emulate_precision_casts),
        "cudnn_benchmark": int(torch.backends.cudnn.benchmark),
        "cudnn_tf32": int(torch.backends.cudnn.allow_tf32),
    }
    with torch.cuda.device(source.device):
        scope = make_launch_scope(
            _product_scan_coarse4_associative_backward_kernel,
            source,
            shape=shape,
            execution_regime="compiled" if compiled else "eager",
        )
        scope = replace(
            scope,
            kernel_revision=_layout_revision(),
            cuda_driver_version=scope.cuda_driver_version or _driver_api_version(),
        )
        selected = validated_geometry(LAYOUT_NAME, scope, BUNDLED_LAYOUTS)
    if selected is not None and selected != BASELINE:
        blocks = selected.blocks
        group = blocks.get("LINES_PER_GROUP", -1)
        tile = blocks.get("BLOCK_MODES", -1)
        compact = blocks.get("COMPACT_CHANNELS", -1)
        try:
            valid = selected == candidate_geometry(
                group, tile, selected.num_warps, compact
            )
        except ValueError:
            valid = False
        if valid and (group == 0 or batch * width % group == 0):
            geometry = LaunchGeometry.build(
                num_warps=selected.num_warps,
                blocks={
                    "BLOCK_LINES": 1,
                    "BLOCK_MODES": tile,
                },
            )
            return group, compact, geometry
    return 0, 0, None


@dataclass(frozen=True)
class _LayoutGraphPass(CustomGraphPass):
    fingerprint: str
    previous: object

    def __call__(self, graph: torch.fx.Graph) -> None:
        if callable(self.previous):
            self.previous(graph)

    def uuid(self):
        if self.previous is None:
            return self.fingerprint
        # Preserve a caller's existing pass and its cacheability.
        previous_id = (
            self.previous.uuid() if isinstance(self.previous, CustomGraphPass) else None
        )
        return (self.fingerprint, previous_id) if previous_id else None


def layout_graph_pass() -> CustomGraphPass:
    """Key the AOT/FX graph on the layout without retuning unrelated Triton kernels."""
    records = geometry_store_signature(LAYOUT_NAME, BUNDLED_LAYOUTS)
    profile = os.environ.get("LNET_COMPILE_PROFILE", "")
    fingerprint = hashlib.sha256(
        f"{_layout_revision()}|{records}|{profile}|{_driver_api_version()}".encode()
    ).hexdigest()
    return _LayoutGraphPass(fingerprint, inductor_config.post_grad_custom_pre_pass)
