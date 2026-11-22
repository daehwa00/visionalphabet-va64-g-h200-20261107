"""Low-rank strict-complex spatial readers for product-scan inputs."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm
from .factorized_reader_kernels import gram_kernel_energy, packed_spatial

ComplexField = tuple[Tensor, Tensor]


class FactorizedComplexConv2dReader(nn.Module):
    """Apply complex pointwise analysis followed by pole-wise spatial filters.

    The effective kernel is

        W[p, k, dy, dx] = sum_r D[p, r, dy, dx] * A[p, r, k],

    where ``A`` and ``D`` are strict-complex weights.  ``rank`` controls how
    many independently detected local events each output pole can combine.
    The VA128 builder fixes execution to ``factorized_triton``. The ``dense``
    and grouped ``factorized`` paths remain available as comparison references.
    All paths preserve the parameter layout, with different floating-point rounding.
    ``factorized_triton`` uses packed pointwise projection and fuses spatial
    filtering, rank reduction and output layout on CUDA, including backward.
    On CPU that mode uses grouped spatial convolution.
    """

    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        rank: int = 2,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
        normalize_input: bool = False,
        match_input_rms: bool = False,
        rms_reference_modes: int | None = None,
        execution_mode: Literal["dense", "factorized", "factorized_triton"] = "dense",
    ) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0:
            message = "factorized complex reader dimensions must be positive"
            raise ValueError(message)
        if rank <= 0:
            message = "factorized complex reader rank must be positive"
            raise ValueError(message)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            message = "factorized complex reader kernel size must be a positive odd integer"
            raise ValueError(message)
        if variance_epsilon <= 0.0:
            message = "factorized complex reader variance epsilon must be positive"
            raise ValueError(message)

        if execution_mode not in ("dense", "factorized", "factorized_triton"):
            raise ValueError("reader execution mode must be dense, factorized or factorized_triton")
        self.execution_mode = execution_mode
        self.input_modes = int(input_modes)
        self.output_modes = int(output_modes)
        self.rank = int(rank)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        self.variance_epsilon = float(variance_epsilon)
        self.normalize_input = bool(normalize_input)
        self.match_input_rms = bool(match_input_rms)
        active_rms_modes = output_modes if rms_reference_modes is None else int(rms_reference_modes)
        if active_rms_modes <= 0 or active_rms_modes > output_modes:
            message = "reader RMS reference modes must be within the output width"
            raise ValueError(message)
        self.rms_reference_modes = active_rms_modes
        self.input_norm: ComplexRMSNorm | None = (
            ComplexRMSNorm(input_modes) if self.normalize_input else None
        )

        point_shape = (output_modes, rank, input_modes)
        spatial_shape = (output_modes, rank, kernel_size, kernel_size)
        self.point_weight_real = nn.Parameter(torch.empty(point_shape))
        self.point_weight_imag = nn.Parameter(torch.empty(point_shape))
        self.spatial_weight_real = nn.Parameter(torch.empty(spatial_shape))
        self.spatial_weight_imag = nn.Parameter(torch.empty(spatial_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use a stable generic initialization for non-square readers."""
        nn.init.xavier_uniform_(self.point_weight_real)
        nn.init.xavier_uniform_(self.point_weight_imag)
        with torch.no_grad():
            scale = math.sqrt(0.5)
            self.point_weight_real.mul_(scale)
            self.point_weight_imag.mul_(scale)
            self.spatial_weight_real.zero_()
            self.spatial_weight_imag.zero_()
            self.spatial_weight_real[:, :, self.padding, self.padding].fill_(
                1.0 / math.sqrt(self.rank)
            )

    def initialize_orthogonal_(self) -> None:
        """Start near identity with active, mutually orthogonal rank components.

        The first component is the identity detector and a centered impulse.
        Every remaining component receives an independent complex point detector
        orthogonal to that identity channel and an independent complex spatial
        filter orthogonal to the centered impulse.  Their total initial energy is
        derived from the input width, so no model shape is special-cased.
        """
        if self.input_modes != self.output_modes:
            message = "orthogonal initialization requires equal input and output modes"
            raise ValueError(message)
        maximum_rank = min(self.input_modes, self.kernel_size * self.kernel_size)
        if self.rank > maximum_rank:
            message = (
                "orthogonal initialization requires rank no greater than both "
                "the input width and spatial support"
            )
            raise ValueError(message)
        with torch.no_grad():
            self.point_weight_real.zero_()
            self.point_weight_imag.zero_()
            self.spatial_weight_real.zero_()
            self.spatial_weight_imag.zero_()
            indices = torch.arange(self.input_modes, device=self.point_weight_real.device)
            self.point_weight_real[indices, 0, indices] = 1.0
            self.spatial_weight_real[:, 0, self.padding, self.padding] = 1.0

            if self.rank == 1:
                return

            residual_rank = self.rank - 1
            point_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    self.input_modes,
                    residual_rank,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    self.input_modes,
                    residual_rank,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
            )
            point_candidates[indices, indices] = 0.0
            point_basis = torch.linalg.qr(point_candidates, mode="reduced").Q.movedim(-1, -2)
            self.point_weight_real[:, 1:].copy_(point_basis.real)
            self.point_weight_imag[:, 1:].copy_(point_basis.imag)

            spatial_size = self.kernel_size * self.kernel_size
            spatial_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    spatial_size,
                    residual_rank,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    spatial_size,
                    residual_rank,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
            )
            center = self.padding * self.kernel_size + self.padding
            spatial_candidates[:, center] = 0.0
            spatial_basis = torch.linalg.qr(spatial_candidates, mode="reduced").Q
            spatial_basis = spatial_basis.movedim(-1, -2).reshape(
                self.output_modes,
                residual_rank,
                self.kernel_size,
                self.kernel_size,
            )
            residual_scale = 1.0 / math.sqrt(self.input_modes)
            self.spatial_weight_real[:, 1:].copy_(spatial_basis.real * residual_scale)
            self.spatial_weight_imag[:, 1:].copy_(spatial_basis.imag * residual_scale)

    def initialize_semi_orthogonal_(self, *, identity_when_square: bool = True) -> None:
        """Initialize a rectangular reader without privileging a fixed width.

        Square readers keep the exact identity-centered initialization above.
        Rectangular readers instead receive the closest available isometry:
        orthonormal rows when compressing and orthonormal columns when
        expanding.  Remaining rank components are independently orthogonal to
        each pole's primary detector and to the centered spatial impulse.

        Set ``identity_when_square=False`` to sample one joint orthogonal square
        analysis basis while retaining the same rank-component contract.
        """
        if self.input_modes == self.output_modes and identity_when_square:
            self.initialize_orthogonal_()
            return
        maximum_rank = min(self.input_modes, self.kernel_size * self.kernel_size)
        if self.rank > maximum_rank:
            message = (
                "semi-orthogonal initialization requires rank no greater than "
                "both the input width and spatial support"
            )
            raise ValueError(message)

        with torch.no_grad():
            self.point_weight_real.zero_()
            self.point_weight_imag.zero_()
            self.spatial_weight_real.zero_()
            self.spatial_weight_imag.zero_()

            primary = torch.empty(
                self.output_modes,
                self.input_modes,
                device=self.point_weight_real.device,
                dtype=self.point_weight_real.dtype,
            )
            nn.init.orthogonal_(primary)
            self.point_weight_real[:, 0].copy_(primary)
            self.spatial_weight_real[:, 0, self.padding, self.padding] = 1.0
            if self.rank == 1:
                return

            residual_rank = self.rank - 1
            primary_complex = torch.complex(primary, torch.zeros_like(primary))
            point_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    self.input_modes,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    self.input_modes,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
            )
            overlap = (primary_complex[:, None].conj() * point_candidates).sum(dim=-1, keepdim=True)
            primary_energy = primary.square().sum(dim=-1, keepdim=True)[:, None]
            point_candidates = point_candidates - (
                overlap * primary_complex[:, None] / primary_energy.clamp_min(1.0e-12)
            )
            point_candidates = point_candidates / torch.linalg.vector_norm(
                point_candidates,
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-12)
            self.point_weight_real[:, 1:].copy_(point_candidates.real)
            self.point_weight_imag[:, 1:].copy_(point_candidates.imag)

            spatial_size = self.kernel_size * self.kernel_size
            spatial_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    spatial_size,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    spatial_size,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
            )
            center = self.padding * self.kernel_size + self.padding
            spatial_candidates[..., center] = 0.0
            spatial_candidates = spatial_candidates / torch.linalg.vector_norm(
                spatial_candidates,
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-12)
            spatial_candidates = spatial_candidates.reshape(
                self.output_modes,
                residual_rank,
                self.kernel_size,
                self.kernel_size,
            )
            residual_scale = 1.0 / math.sqrt(self.input_modes)
            self.spatial_weight_real[:, 1:].copy_(spatial_candidates.real * residual_scale)
            self.spatial_weight_imag[:, 1:].copy_(spatial_candidates.imag * residual_scale)

    def synthesized_kernel(self) -> ComplexField:
        """Synthesize the strict-complex K-by-K kernel represented by the factors."""
        point_real = self.point_weight_real.float()
        point_imag = self.point_weight_imag.float()
        spatial_real = self.spatial_weight_real.float()
        spatial_imag = self.spatial_weight_imag.float()
        full_real = torch.einsum("prij,prk->pkij", spatial_real, point_real)
        full_real = full_real - torch.einsum("prij,prk->pkij", spatial_imag, point_imag)
        full_imag = torch.einsum("prij,prk->pkij", spatial_real, point_imag)
        full_imag = full_imag + torch.einsum("prij,prk->pkij", spatial_imag, point_real)
        return full_real, full_imag

    def unit_energy_kernel(self) -> ComplexField:
        """Return the synthesized kernel with exact unit row energy."""
        full_real, full_imag = self.synthesized_kernel()
        inverse_rms = torch.rsqrt(
            full_real.square()
            .add(full_imag.square())
            .sum(dim=(1, 2, 3), keepdim=True)
            .clamp_min(self.variance_epsilon)
        )
        return full_real * inverse_rms, full_imag * inverse_rms

    def _validate_input(self, real: Tensor, imag: Tensor) -> None:
        if real.shape != imag.shape or real.ndim != 4:
            message = "factorized complex reader requires matching NHWM fields"
            raise ValueError(message)
        if real.shape[-1] != self.input_modes:
            message = "factorized complex reader input has an incompatible mode dimension"
            raise ValueError(message)

    def _factorized_conv(self, packed_input: Tensor) -> Tensor:
        """Apply the same normalized kernel without a dense spatial convolution.

        Projection channels are (component, pole, rank). Grouped spatial channels
        are (pole, component, rank), with one complex output per pole. Normalize
        the spatial factors using the *full* synthesized row energy, preserving
        cross-rank terms and gradients through the normalization denominator.
        """
        point_real = self.point_weight_real.float().flatten(0, 1)[..., None, None]
        point_imag = self.point_weight_imag.float().flatten(0, 1)[..., None, None]
        point_kernel = torch.cat((
            torch.cat((point_real, -point_imag), dim=1),
            torch.cat((point_imag, point_real), dim=1),
        ), dim=0)
        projected = functional.conv2d(packed_input, point_kernel)
        batch, _, height, width = projected.shape
        grouped_input = projected.reshape(
            batch, 2, self.output_modes, self.rank, height, width,
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, 2 * self.output_modes * self.rank, height, width,
        )
        full_real, full_imag = self.synthesized_kernel()
        inverse_rms = torch.rsqrt(
            full_real.square().add(full_imag.square()).sum(dim=(1, 2, 3), keepdim=True)
            .clamp_min(self.variance_epsilon)
        )
        spatial_real = self.spatial_weight_real.float() * inverse_rms
        spatial_imag = self.spatial_weight_imag.float() * inverse_rms
        spatial_kernel = torch.stack((
            torch.cat((spatial_real, -spatial_imag), dim=1),
            torch.cat((spatial_imag, spatial_real), dim=1),
        ), dim=1).reshape(
            2 * self.output_modes, 2 * self.rank, self.kernel_size, self.kernel_size,
        )
        grouped_output = functional.conv2d(
            grouped_input, spatial_kernel, padding=self.padding, groups=self.output_modes,
        )
        return grouped_output.reshape(batch, self.output_modes, 2, height, width).permute(
            0, 2, 1, 3, 4,
        ).reshape(batch, 2 * self.output_modes, height, width)

    def _pointwise_kernel(self) -> Tensor:
        point_real = self.point_weight_real.float().transpose(0, 1).flatten(0, 1)
        point_imag = self.point_weight_imag.float().transpose(0, 1).flatten(0, 1)
        return torch.cat((
            torch.cat((point_real, -point_imag), dim=1),
            torch.cat((point_imag, point_real), dim=1),
        ), dim=0)

    def _normalized_spatial_factors(self) -> ComplexField:
        energy = gram_kernel_energy(
            self.point_weight_real, self.point_weight_imag,
            self.spatial_weight_real, self.spatial_weight_imag,
        )
        inverse_rms = torch.rsqrt(energy.clamp_min(self.variance_epsilon))[:, None, None, None]
        return (
            self.spatial_weight_real.float() * inverse_rms,
            self.spatial_weight_imag.float() * inverse_rms,
        )

    def _factorized_triton_conv_packed(self, packed_input: Tensor) -> Tensor:
        projected = functional.linear(packed_input, self._pointwise_kernel().to(packed_input.dtype))
        return packed_spatial(projected, *self._normalized_spatial_factors())

    def forward_packed(self, packed: Tensor) -> Tensor:
        """Read NHW[real | imag] coordinates without unpacking activation buffers.

        CUDA factorized execution projects packed rows directly and writes one
        packed spatial output. Parameters and normalization equations are shared
        with the separate-coordinate API, preserving checkpoint compatibility.
        """
        if packed.ndim != 4 or packed.shape[-1] != 2 * self.input_modes:
            raise ValueError("factorized complex reader requires NHW[real | imag] input")
        if self.input_norm is not None:
            # The reader's reference normalizer preserves its input dtype. In
            # particular RMS matching must see FP32 normalized rows, not an
            # early BF16 cast introduced by the PostFusion-specific kernel.
            real, imag = packed.chunk(2, dim=-1)
            energy = (real.float().square() + imag.float().square()).mean(-1, keepdim=True)
            inverse_rms = torch.rsqrt(energy + self.input_norm.epsilon).to(packed.dtype)
            weight = self.input_norm.weight.to(packed.dtype)
            if self.execution_mode == "factorized_triton":
                packed = packed * inverse_rms * weight.repeat(2)
            else:
                coordinates = packed.unflatten(-1, (2, self.input_modes))
                packed = (coordinates * inverse_rms.unsqueeze(-1) * weight).flatten(-2)

        return self._read_packed(packed)

    def _read_packed(self, packed: Tensor) -> Tensor:
        if self.execution_mode == "factorized_triton":
            output = self._factorized_triton_conv_packed(packed)
        elif self.execution_mode == "factorized":
            output = self._factorized_conv(packed.movedim(-1, 1)).movedim(1, -1).contiguous()
        elif self.execution_mode == "dense":
            kernel_real, kernel_imag = self.unit_energy_kernel()
            kernel = torch.cat((
                torch.cat((kernel_real, -kernel_imag), dim=1),
                torch.cat((kernel_imag, kernel_real), dim=1),
            ), dim=0)
            output = functional.conv2d(
                packed.movedim(-1, 1), kernel, padding=self.padding,
            ).movedim(1, -1).contiguous()
        else:
            raise ValueError("reader execution mode must be dense, factorized or factorized_triton")

        if self.match_input_rms:
            input_energy = packed.float().square().sum(dim=-1, keepdim=True).div(self.input_modes)
            coordinates = output.reshape(*output.shape[:-1], 2, self.output_modes)
            output_energy = (
                coordinates[..., 0, : self.rms_reference_modes].float().square().sum(dim=-1)
                + coordinates[..., 1, : self.rms_reference_modes].float().square().sum(dim=-1)
            ).unsqueeze(-1).div(self.rms_reference_modes
            )
            scale = torch.sqrt(
                (input_energy + self.variance_epsilon) / (output_energy + self.variance_epsilon)
            ).to(output.dtype)
            output = output * scale
        return output

    def forward(self, real: Tensor, imag: Tensor | None = None) -> ComplexField | Tensor:
        if imag is None:
            return self.forward_packed(real)
        self._validate_input(real, imag)
        if self.input_norm is not None:
            real, imag = self.input_norm(real, imag)
        if self.execution_mode == "factorized_triton":
            output = self._read_packed(torch.cat((real, imag), dim=-1))
            out_real, out_imag = output.chunk(2, dim=-1)
            return out_real.contiguous(), out_imag.contiguous()
        packed_input = torch.cat(
            (real.movedim(-1, 1), imag.movedim(-1, 1)), dim=1,
        )
        if self.execution_mode == "dense":
            kernel_real, kernel_imag = self.unit_energy_kernel()
            kernel_top = torch.cat((kernel_real, -kernel_imag), dim=1)
            kernel_bottom = torch.cat((kernel_imag, kernel_real), dim=1)
            packed_kernel = torch.cat((kernel_top, kernel_bottom), dim=0)
            output = functional.conv2d(packed_input, packed_kernel, padding=self.padding)
        elif self.execution_mode == "factorized":
            output = self._factorized_conv(packed_input)
        else:
            raise ValueError("reader execution mode must be dense, factorized or factorized_triton")

        if self.match_input_rms:
            input_energy = (
                packed_input.float().square().sum(dim=1, keepdim=True).div(self.input_modes)
            )
            output_energy = (
                output.float()
                .reshape(
                    output.shape[0],
                    2,
                    self.output_modes,
                    output.shape[2],
                    output.shape[3],
                )
                [:, :, : self.rms_reference_modes]
                .square()
                .sum(dim=(1, 2), keepdim=False)
                .unsqueeze(1)
                .div(
                    self.rms_reference_modes,
                )
            )
            token_scale = torch.sqrt(
                (input_energy + self.variance_epsilon) / (output_energy + self.variance_epsilon)
            ).to(dtype=output.dtype)
            output = output * token_scale

        return (
            output[:, : self.output_modes].movedim(1, -1).contiguous(),
            output[:, self.output_modes :].movedim(1, -1).contiguous(),
        )


__all__ = ["FactorizedComplexConv2dReader"]
