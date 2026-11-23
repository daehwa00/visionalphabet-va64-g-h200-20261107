"""Shared image stems, normalization, and descriptor heads."""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn
from torch.nn import functional

from .fused_stem_ops import fused_nhwc_layernorm_gelu

_FUSED_FACTORIZED_READER_ENV = "LNET_FUSED_FACTORIZED_READER"
_GENERAL_SCANS_ENV = "LNET_GENERAL_SCANS"
_FUSED_STEM_MIN_ELEMENTS = 8 * 1024 * 1024


class LayerNorm2d(nn.Module):
    """Apply channel-wise LayerNorm to an NCHW feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return (
            self.norm(inputs.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous(memory_format=torch.channels_last)
        )


class CifarConvStem(nn.Module):
    """Two overlapping convolutions with an explicit dataset-scale stride."""

    def __init__(
        self,
        output_width: int = 64,
        strides: tuple[int, int] = (1, 1),
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        first = nn.Conv2d(3, 32, 3, stride=strides[0], padding=1, bias=False)
        second = nn.Conv2d(
            32,
            output_width,
            3,
            stride=strides[1],
            padding=1,
            bias=False,
        )
        if bias:
            # Preserve the bias-free initialization while making the offsets trainable.
            first.bias = nn.Parameter(torch.zeros(32))
            second.bias = nn.Parameter(torch.zeros(output_width))
        self.layers = nn.Sequential(
            first,
            LayerNorm2d(32),
            nn.GELU(),
            second,
            LayerNorm2d(output_width),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.is_cuda and os.environ.get("LNET_DIRECT_CONV_STEM") == "on":
            hidden = self._try_fused_conv_stem(inputs)
            if hidden is not None:
                return hidden.permute(0, 2, 3, 1)
        return self.layers(inputs).permute(0, 2, 3, 1)

    def _try_fused_conv_stem(self, inputs: Tensor) -> Tensor | None:
        first = self.layers[0]
        if not (
            type(first) is nn.Conv2d
            and first.kernel_size == (3, 3)
            and first.stride == (2, 2)
            and first.padding == (1, 1)
            and first.dilation == (1, 1)
            and first.groups == 1
            and first.padding_mode == "zeros"
        ):
            return None
        from .fused_conv_stem_ops import try_direct_conv3x3s2  # noqa: PLC0415

        hidden = try_direct_conv3x3s2(inputs, first.weight, first.bias)
        if hidden is None:
            return None
        for layer in list(self.layers)[1:]:
            hidden = layer(hidden)
        return hidden


def _fused_stem_enabled(inputs: Tensor) -> bool:
    return inputs.is_cuda and (
        os.environ.get(_FUSED_FACTORIZED_READER_ENV) != "0"
        or os.environ.get(_GENERAL_SCANS_ENV) != "0"
    )


def _supports_fused_stem_block(conv: nn.Module, norm: nn.Module, gelu: nn.Module) -> bool:
    return (
        type(conv) is nn.Conv2d
        and conv.padding_mode == "zeros"
        and type(norm) is LayerNorm2d
        and type(norm.norm) is nn.LayerNorm
        and norm.norm.weight is not None
        and norm.norm.bias is not None
        and norm.norm.normalized_shape == (conv.out_channels,)
        and type(gelu) is nn.GELU
        and gelu.approximate == "none"
    )


def _conv2d_output_dim(size: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _fused_stem_large_enough(inputs: Tensor, first: nn.Conv2d, second: nn.Conv2d) -> bool:
    if inputs.ndim != 4:
        return False
    h1 = _conv2d_output_dim(
        inputs.shape[2],
        int(first.padding[0]),
        first.dilation[0],
        first.kernel_size[0],
        first.stride[0],
    )
    w1 = _conv2d_output_dim(
        inputs.shape[3],
        int(first.padding[1]),
        first.dilation[1],
        first.kernel_size[1],
        first.stride[1],
    )
    h2 = _conv2d_output_dim(
        h1,
        int(second.padding[0]),
        second.dilation[0],
        second.kernel_size[0],
        second.stride[0],
    )
    w2 = _conv2d_output_dim(
        w1,
        int(second.padding[1]),
        second.dilation[1],
        second.kernel_size[1],
        second.stride[1],
    )
    return inputs.shape[0] * second.out_channels * h2 * w2 >= _FUSED_STEM_MIN_ELEMENTS


def _fused_stem_block(
    inputs: Tensor,
    conv: nn.Conv2d,
    norm: LayerNorm2d,
    *,
    nhwc_output: bool = False,
) -> Tensor:
    hidden = functional.conv2d(
        inputs, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups
    )
    return fused_nhwc_layernorm_gelu(
        hidden,
        norm.norm.weight,
        norm.norm.bias,
        norm.norm.eps,
        nchw=True,
        conv_bias=conv.bias,
        output_dtype=hidden.dtype,
        nhwc_output=nhwc_output,
    )


class ModeScaledTwoConvStem(nn.Module):
    """Apply 3-to-H-to-2K Conv-LN-GELU blocks and return NHWC features."""

    def __init__(
        self,
        modes: int,
        strides: tuple[int, int] = (2, 2),
        *,
        input_channels: int = 3,
        hidden_width: int = 32,
    ) -> None:
        super().__init__()
        if modes <= 0 or hidden_width <= 0 or input_channels <= 0:
            message = "mode-scaled stem dimensions must be positive"
            raise ValueError(message)
        if len(strides) != 2 or any(stride <= 0 for stride in strides):
            message = "mode-scaled stem requires two positive strides"
            raise ValueError(message)
        self.hidden_width = hidden_width
        self.output_width = 2 * modes
        first = nn.Conv2d(
            input_channels,
            hidden_width,
            3,
            stride=strides[0],
            padding=1,
            bias=True,
        )
        second = nn.Conv2d(
            hidden_width,
            self.output_width,
            3,
            stride=strides[1],
            padding=1,
            bias=True,
        )
        if first.bias is None or second.bias is None:
            raise RuntimeError("mode-scaled stem requires affine convolution biases")
        nn.init.zeros_(first.bias)
        nn.init.zeros_(second.bias)
        self.layers = nn.Sequential(
            first,
            LayerNorm2d(hidden_width),
            nn.GELU(),
            second,
            LayerNorm2d(self.output_width),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        first, first_norm, first_activation, second, second_norm, second_activation = self.layers
        if (
            _fused_stem_enabled(inputs)
            and _supports_fused_stem_block(first, first_norm, first_activation)
            and _supports_fused_stem_block(second, second_norm, second_activation)
            and _fused_stem_large_enough(inputs, first, second)
        ):
            hidden = _fused_stem_block(inputs, first, first_norm)
            return _fused_stem_block(hidden, second, second_norm, nhwc_output=True)
        hidden = first_activation(first_norm(first(inputs)))
        features = second(hidden)
        if not isinstance(second_norm, LayerNorm2d):
            message = "mode-scaled stem requires a terminal LayerNorm2d"
            raise TypeError(message)
        features = second_norm.norm(features.permute(0, 2, 3, 1))
        return second_activation(features)


class ResidualPreComplexMixer(nn.Module):
    """Turn the established square two-linear path into a residual mixer."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if not isinstance(source, nn.Sequential) or len(source) != 4:
            message = "residual pre-complex mixer requires the established two-linear path"
            raise TypeError(message)
        input_projection, activation, output_projection, tail = tuple(source.children())
        if not isinstance(input_projection, nn.Linear) or not isinstance(
            output_projection,
            nn.Linear,
        ):
            message = "residual pre-complex mixer requires two linear projections"
            raise TypeError(message)
        width = input_projection.in_features
        if (
            not isinstance(activation, nn.GELU)
            or not isinstance(tail, nn.Identity)
            or input_projection.out_features != width
            or output_projection.in_features != width
            or output_projection.out_features != width
        ):
            message = "residual pre-complex mixer requires equal input and output widths"
            raise TypeError(message)
        self.width = width
        self.input_projection = input_projection
        self.activation = activation
        self.output_projection = output_projection

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.activation(self.input_projection(inputs))
        flat_inputs = inputs.reshape(-1, self.width)
        flat_update = update.reshape(-1, self.width)
        mixed = torch.addmm(
            flat_inputs,
            flat_update,
            self.output_projection.weight.T,
        )
        if self.output_projection.bias is not None:
            biased = mixed + self.output_projection.bias
            # Keep the residual projection in the low-precision stem's dtype.
            if inputs.dtype in (torch.float16, torch.bfloat16) and _fused_stem_enabled(inputs):
                biased = biased.to(inputs.dtype)
            mixed = biased
        return mixed.reshape_as(inputs)


class LowRankQuadraticModalHead(nn.Module):
    """Structured modal classifier with linear and rank-wise attribution."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or rank <= 0:
            message = "modal head dimensions and rank must be positive"
            raise ValueError(message)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.linear = nn.Linear(input_dim, output_dim)
        self.projection = nn.Linear(input_dim, rank, bias=False)
        self.quadratic = nn.Linear(rank, output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        nn.init.normal_(self.quadratic.weight, std=1.0e-3)

    def standardized(self, descriptor: Tensor) -> Tensor:
        return self.standardizer(descriptor)

    def forward(self, descriptor: Tensor) -> Tensor:
        standardized = self.standardized(descriptor)
        interaction = self.projection(standardized).square()
        return self.linear(standardized) + self.quadratic(interaction)

    def decompose(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return bias, coordinate-linear, and latent-quadratic logit terms."""
        standardized = self.standardized(descriptor)
        linear = standardized[:, None, :] * self.linear.weight[None, :, :]
        latent = self.projection(standardized).square()
        quadratic = latent[:, None, :] * self.quadratic.weight[None, :, :]
        bias = self.linear.bias
        if bias is None:
            message = "the modal linear term requires a bias"
            raise RuntimeError(message)
        return bias, linear, quadratic


class LivePolewiseComplexRMS(nn.Module):
    """Normalize paired real and imaginary coordinates with one scale per pole."""

    def __init__(
        self,
        input_dim: int,
        *,
        momentum: float = 0.05,
        eps: float = 1e-12,
        output_scale: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or input_dim % 2:
            raise ValueError("complex RMS requires a positive even input_dim")
        self.input_dim = input_dim
        self.momentum = momentum
        self.eps = eps
        self.output_scale = output_scale
        self.center_mean = False
        self.complex_modulus = True
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_ms", torch.zeros(input_dim // 2))
        self.register_buffer("ema_weight", torch.zeros(()))

    def eval_pole_scales(self) -> Tensor:
        w_safe = self.ema_weight.float().clamp_min(1e-12)
        ms_est = (self.running_ms.float() / w_safe).clamp_min(0.0)
        return self.output_scale * torch.rsqrt(ms_est + self.eps * self.eps)

    def eval_full_scales(self) -> Tensor:
        scales = self.eval_pole_scales()
        return torch.cat((scales, scales))

    def forward(self, descriptor: Tensor) -> Tensor:
        if descriptor.ndim < 2 or descriptor.shape[-1] != self.input_dim:
            raise ValueError("complex RMS requires a batched descriptor with matching input_dim")
        x = descriptor.float()
        x2 = x.reshape(*x.shape[:-1], 2, self.input_dim // 2)
        if self.training:
            reduce_dims = tuple(range(x2.ndim - 2))
            batch_ms = x2.square().sum(dim=-2).mean(dim=reduce_dims)
            with torch.no_grad():
                m = self.momentum
                w_new = self.ema_weight * (1.0 - m) + m
                self.running_ms.mul_(1.0 - m).add_(batch_ms.detach(), alpha=m)
                self.ema_weight.copy_(w_new)
            inv_rms = self.output_scale * torch.rsqrt(batch_ms + self.eps * self.eps)
        else:
            inv_rms = self.eval_pole_scales()
        return (x2 * inv_rms.unsqueeze(-2)).reshape_as(x)


class StandardizedAffineModalHead(nn.Module):
    """Affine modal classifier with parameter-free standardization."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.folded_standardizer = False
        self.standardizer: nn.Module
        if input_dim >= 256 and input_dim % 2 == 0:
            self.standardizer = LivePolewiseComplexRMS(input_dim)
        else:
            self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.linear = nn.Linear(input_dim, output_dim)

    @torch.no_grad()
    def fold_standardizer_into_linear(self) -> StandardizedAffineModalHead:
        if isinstance(self.standardizer, LivePolewiseComplexRMS):
            scale = self.standardizer.eval_full_scales()
            weight = (self.linear.weight.float() * scale.unsqueeze(0)).to(self.linear.weight.dtype)
            self.linear.weight.copy_(weight)
            self.standardizer = nn.Identity().train(self.training)
            self.folded_standardizer = True
        return self

    def forward(self, descriptor: Tensor) -> Tensor:
        if self.folded_standardizer and self.training:
            raise RuntimeError("cannot run folded StandardizedAffineModalHead in training mode")
        return self.linear(self.standardizer(descriptor))


def fold_terminal_rms_for_inference(model: nn.Module) -> nn.Module:
    """Fold live RMS heads in place using their frozen evaluation scales."""
    for module in list(model.modules()):
        if isinstance(module, StandardizedAffineModalHead):
            module.fold_standardizer_into_linear()
    return model


__all__ = [
    "CifarConvStem",
    "LayerNorm2d",
    "LivePolewiseComplexRMS",
    "LowRankQuadraticModalHead",
    "ModeScaledTwoConvStem",
    "ResidualPreComplexMixer",
    "StandardizedAffineModalHead",
    "fold_terminal_rms_for_inference",
]
