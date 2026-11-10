"""Canonical head-wise LRCP module.

Model- or task-specific files should only adapt tensor wiring, layer selection,
optimizer parameter grouping, and logging.  The LRCP branch itself lives here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from .functional import apple_bshd, project_bshd


__all__ = ["LRCP"]


class LRCP(nn.Module):
    """Low-rank complement branch for head-split tensors shaped ``[B, H, N, D]``."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        rank: int,
        gamma: float,
        down_init_std: float = 0.02,
        up_init_std: float = 0.0,
        residual_detach: bool = False,
        *,
        telemetry_prefix: str = "",
    ):
        super().__init__()
        if isinstance(residual_detach, str):
            raise TypeError(
                "LRCP no longer accepts an implementation/mode positional argument; "
                "use the canonical LRCP branch and pass residual_detach as a bool."
            )
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.gamma = float(gamma)
        self.residual_detach = bool(residual_detach)
        self.telemetry_prefix = telemetry_prefix
        self.stats_enabled = False
        # Optional fused paths require explicit enablement. The shared
        # dispatcher keeps CPU, telemetry and unsupported inputs on PyTorch.
        self.use_fused_inference = False
        self.use_fused_training = False
        self.last_stats: dict[str, float] = {}
        self.last_telemetry: dict[str, float] = {}
        self.down = nn.Parameter(torch.empty(num_heads, head_dim, rank))
        self.up = nn.Parameter(torch.empty(num_heads, rank, head_dim))
        nn.init.normal_(self.down, mean=0.0, std=down_init_std)
        if up_init_std == 0.0:
            nn.init.zeros_(self.up)
        else:
            nn.init.normal_(self.up, mean=0.0, std=up_init_std)

    def _project_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(residual, self.down), self.up)

    def _project_residual_bshd(self, residual: torch.Tensor) -> torch.Tensor:
        """Project a ``[B,S,H,D]`` residual without materializing a transpose."""

        return project_bshd(residual, self.down, self.up)

    def branch_from_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Return the LRCP branch for an already-formed residual tensor.

        This small public probe API keeps analysis scripts from reimplementing
        the internal down/up projection math.  It is intentionally branch-only:
        callers that need the normal module behavior should use ``forward``.
        """

        if residual.ndim != 4:
            raise ValueError(f"LRCP residual expects [B,H,N,D], got {tuple(residual.shape)}")
        if residual.shape[1] != self.num_heads or residual.shape[-1] != self.head_dim:
            raise ValueError(
                f"residual shape {tuple(residual.shape)} does not match heads={self.num_heads}, dim={self.head_dim}"
            )
        return self._project_residual(residual) * self.gamma

    def branch(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Return the LRCP branch computed from ``value_states - av``."""

        self._validate_inputs(av, value_states)
        return self.branch_from_residual(self._residual(av, value_states))

    def forward(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(av, value_states)
        residual = self._residual(av, value_states)
        raw = self._project_residual(residual)
        branch = raw if self.gamma == 1.0 else raw * self.gamma
        if self.stats_enabled:
            self._record_stats(av=av, residual=residual, raw=raw, branch=branch)
        return av + branch

    def forward_bshd(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Layout-aware forward for attention kernels returning ``[B,S,H,D]``.

        This is mathematically identical to :meth:`forward`, but avoids the two
        contiguous layout copies otherwise needed around the canonical BHSD
        interface.
        """

        self._validate_inputs_bshd(av, value_states)
        if not self.stats_enabled:
            fused_requested = (
                self.use_fused_training and self.training and torch.is_grad_enabled()
            ) or (
                self.use_fused_inference and not self.training and not torch.is_grad_enabled()
            )
            return apple_bshd(
                av, value_states, self.down, self.up, self.gamma,
                backend="triton" if fused_requested else "torch",
                residual_detach=self.residual_detach,
            )
        residual = self._residual_bshd(av, value_states)
        raw = self._project_residual_bshd(residual)
        branch = raw if self.gamma == 1.0 else raw * self.gamma
        if self.stats_enabled:
            self._record_stats(av=av, residual=residual, raw=raw, branch=branch)
        return av + branch

    def _residual_bshd(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        residual = value_states - av
        if self.residual_detach:
            residual = residual.detach()
        return residual

    def _validate_inputs(self, av: torch.Tensor, value_states: torch.Tensor) -> None:
        if av.ndim != 4 or value_states.ndim != 4:
            raise ValueError(f"LRCP expects [B,H,N,D] tensors, got {tuple(av.shape)} and {tuple(value_states.shape)}")
        if av.shape[1] != self.num_heads or av.shape[-1] != self.head_dim:
            raise ValueError(f"av shape {tuple(av.shape)} does not match heads={self.num_heads}, dim={self.head_dim}")
        if value_states.shape[1] != self.num_heads or value_states.shape[-1] != self.head_dim:
            raise ValueError(
                f"value_states shape {tuple(value_states.shape)} does not match heads={self.num_heads}, dim={self.head_dim}"
            )

    def _validate_inputs_bshd(self, av: torch.Tensor, value_states: torch.Tensor) -> None:
        if av.ndim != 4 or value_states.ndim != 4:
            raise ValueError(f"LRCP expects [B,S,H,D] tensors, got {tuple(av.shape)} and {tuple(value_states.shape)}")
        if av.shape != value_states.shape:
            raise ValueError(f"LRCP requires matching AV/value shapes, got {tuple(av.shape)} and {tuple(value_states.shape)}")
        if av.shape[2] != self.num_heads or av.shape[-1] != self.head_dim:
            raise ValueError(f"av shape {tuple(av.shape)} does not match heads={self.num_heads}, dim={self.head_dim}")

    def _residual(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        if value_states.shape[0] != av.shape[0] or value_states.shape[-2] != av.shape[-2]:
            raise ValueError(
                "LRCP residual requires matching batch and token/ray length; "
                f"got av={tuple(av.shape)}, value_states={tuple(value_states.shape)}"
            )
        residual = value_states - av
        if self.residual_detach:
            residual = residual.detach()
        return residual

    def _record_stats(
        self,
        *,
        av: torch.Tensor,
        residual: torch.Tensor,
        raw: torch.Tensor,
        branch: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            eps = 1e-12
            av_norm = av.detach().float().norm().clamp_min(eps)
            prefix = self.telemetry_prefix
            stats = {
                f"{prefix}branch_over_av": float((branch.detach().float().norm() / av_norm).cpu()),
                f"{prefix}raw_branch_over_av": float((raw.detach().float().norm() / av_norm).cpu()),
                f"{prefix}residual_over_av": float((residual.detach().float().norm() / av_norm).cpu()),
                f"{prefix}gamma_mean": self.gamma,
            }
        self.last_stats = stats
        self.last_telemetry = stats
