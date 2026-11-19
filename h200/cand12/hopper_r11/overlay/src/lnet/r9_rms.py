"""Opt-in FP32 RMSNorm scale-gradient reduction for the dense 28² stage."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx import Graph, Node
from torch.library import triton_op, wrap_triton

from .kernel_launch_config import LaunchGeometry, autotuned


@triton.jit
def r9_channel_weight_partial(
    DY,
    X,
    INV,
    DW,
    N: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    SX: tl.constexpr,
    SI: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    channels = tl.program_id(0) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    channel = channels[None, :]
    row = tl.program_id(1) * 128 + tl.arange(0, ROW_TILE)[:, None]
    batch = row // (H * H)
    y, x = (row // H) % H, row % H
    active = (row < N * H * H) & (tl.arange(0, ROW_TILE)[:, None] < 128)
    valid = active & (channel < C)
    gradient = tl.load(DY + row * C + channel, valid, 0).to(tl.float32)
    value = tl.load(
        X + batch * SX[0] + y * SX[1] + x * SX[2] + channel * SX[3],
        valid,
        0,
    ).to(tl.float32)
    inverse = tl.load(INV + batch * SI[0] + y * SI[1] + x * SI[2], active, 0)
    scale_gradient = tl.sum(gradient * (value * inverse), 0)
    tl.store(DW + tl.program_id(1) * C + channels, scale_gradient, channels < C)


@triton_op("lnet::r9_rms_scale_gradient", mutates_args={})
def _rms_scale_gradient(gradient: Tensor, values: Tensor, inverse: Tensor) -> Tensor:
    batch, height, width, channels = values.shape
    partial_rows = triton.cdiv(batch * height * width, 128)
    partial = torch.empty((partial_rows, channels), dtype=torch.float32, device=values.device)
    geometry = LaunchGeometry.build(num_warps=4, blocks={"CHANNEL_TILE": 16, "ROW_TILE": 512})
    kernel = autotuned(
        r9_channel_weight_partial, "r9_rms_scale_gradient", key=(), geometry=geometry
    )
    wrap_triton(kernel)[(triton.cdiv(channels, 16), partial_rows)](
        gradient.contiguous(),
        values,
        inverse,
        partial,
        N=batch,
        H=height,
        C=channels,
        SX=values.stride(),
        SI=inverse.stride(),
    )
    return partial.sum(0).view(1, 1, 1, channels)


def _tensor_value(node: object) -> Tensor | None:
    if isinstance(node, Node):
        value = node.meta.get("val")
        if isinstance(value, Tensor):
            return value
    return None


# BEGIN R10 DISPATCH
# R10 LEGACY: def _rms_operands(node: Node) -> tuple[Node, Node, Node] | None:
def _rms_operands(
    node: Node,
    *,
    shape=(28, 28, 96),
    value_stride=(75264, 1, 2688, 28),
    row_tile=128,
    allow_matching_grad_stride=False,
) -> tuple[Node, Node, Node] | None:
# END R10 DISPATCH
    if node.target != torch.ops.aten.sum.dim_IntList:
        return None
    if len(node.args) != 3:
        return None
    _, dims, keepdim = node.args
    if not isinstance(dims, (list, tuple)) or tuple(dims) != (0, 1, 2) or keepdim is not True:
        return None
    if node.kwargs.get("dtype") != torch.float32:
        return None
    product = node.args[0]
    if not isinstance(product, Node) or product.target != torch.ops.aten.mul.Tensor:
        return None
    for gradient, unit in (product.args, product.args[::-1]):
        if not isinstance(unit, Node) or unit.target != torch.ops.aten.mul.Tensor:
            continue
        for source, cast in (unit.args, unit.args[::-1]):
            if (
                not isinstance(cast, Node)
                or cast.target != torch.ops.prims.convert_element_type.default
            ):
                continue
            inverse, dtype = cast.args
            if dtype != torch.bfloat16 or not isinstance(inverse, Node):
                continue
            # AOTAutograd passes the saved rsqrt from the forward graph as a placeholder.
            if inverse.op != "placeholder" and inverse.target != torch.ops.aten.rsqrt.default:
                continue
            dy, x, inv = (_tensor_value(v) for v in (gradient, source, inverse))
            if dy is None or x is None or inv is None:
                continue
            # BEGIN R10 DISPATCH
            # R10 LEGACY:             if not x.is_cuda or x.ndim != 4 or x.shape[1:] != (28, 28, 96):
            if not x.is_cuda or x.ndim != 4 or x.shape[1:] != shape:
            # END R10 DISPATCH
                continue
            batch = x.shape[0]
            # BEGIN R10 DISPATCH
            # R10 LEGACY:             if not isinstance(batch, int) or not 0 < batch * 28 * 28 * 96 < 2**31:
            height, width, channels = shape
            if not isinstance(batch, int) or not 0 < batch * height * width * channels < 2**31:
            # END R10 DISPATCH
                continue
            # BEGIN R10 DISPATCH
            # R10 LEGACY:             if triton.cdiv(batch * 28 * 28, 128) > 65535:
            if triton.cdiv(batch * height * width, row_tile) > 65535:
            # END R10 DISPATCH
                continue
            if (
                dy.shape != x.shape
                or dy.dtype != x.dtype
                or x.dtype != torch.bfloat16
                # BEGIN R10 DISPATCH
                # R10 LEGACY:                 or not dy.is_contiguous()
                # R10 LEGACY:                 or x.stride() != (75264, 1, 2688, 28)
                or not (
                    dy.is_contiguous()
                    or allow_matching_grad_stride and dy.stride() == value_stride
                )
                or x.stride() != value_stride
                # END R10 DISPATCH
            ):
                continue
            if inv.shape != (*x.shape[:-1], 1) or inv.dtype != torch.float32:
                continue
            # BEGIN R10 DISPATCH
            if allow_matching_grad_stride and sum(
                (size - 1) * stride for size, stride in zip(inv.shape, inv.stride(), strict=True)
            ) >= 2**31:
                continue
            # END R10 DISPATCH
            return gradient, source, inverse
    return None


@dataclass(frozen=True)
class RMSScaleGradientPass(CustomGraphPass):
    previous: object

    def __call__(self, graph: Graph) -> None:
        if callable(self.previous):
            self.previous(graph)
        if config.emulate_precision_casts:
            return
        for node in list(graph.nodes):
            operands = _rms_operands(node)
            if operands is None:
                continue
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    torch.ops.lnet.r9_rms_scale_gradient.default, operands
                )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

    def uuid(self):
        if self.previous is None:
            return "r9_rms_scale_gradient"
        if isinstance(self.previous, CustomGraphPass):
            key = self.previous.uuid()
            if key is not None:
                return ("r9_rms_scale_gradient", key)
        return None
