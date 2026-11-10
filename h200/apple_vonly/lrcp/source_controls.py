"""Parameter-matched controls for testing which attention signal matters.

The canonical LRCP module always consumes ``V - AV``. These controls keep the
same head-wise low-rank operator and initialization, but replace only its input
with ``AV``, ``V``, or a position-permuted ``V - AV``. They live outside the
canonical implementation so ablations cannot silently change the main LRCP.
"""

from __future__ import annotations

import torch

from .headwise import LRCP


__all__ = ["LRCPSourceControl", "replace_lrcp_source"]


class LRCPSourceControl(LRCP):
    """An LRCP-shaped branch whose input source is an experimental control."""

    VALID_SOURCES = {"av", "value", "shuffled_residual"}

    def __init__(self, *args, source: str, **kwargs):
        super().__init__(*args, **kwargs)
        if source not in self.VALID_SOURCES:
            raise ValueError(f"Unknown source control {source!r}; expected one of {sorted(self.VALID_SOURCES)}")
        self.source = source

    def forward_bshd(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        # Canonical apple_bshd hard-codes V-AV, even for its torch backend.
        # Source controls must never enter that dispatch path.
        self._validate_inputs_bshd(av, value_states)
        residual = self._residual_bshd(av, value_states)
        raw = self._project_residual_bshd(residual)
        branch = raw if self.gamma == 1.0 else raw * self.gamma
        if self.stats_enabled:
            self._record_stats(av=av, residual=residual, raw=raw, branch=branch)
        return av + branch

    def _residual(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(av, value_states)
        if value_states.shape[0] != av.shape[0] or value_states.shape[-2] != av.shape[-2]:
            raise ValueError(
                "Source control requires matching batch and token/ray length; "
                f"got av={tuple(av.shape)}, value_states={tuple(value_states.shape)}"
            )
        if self.source == "av":
            branch_input = av
        elif self.source == "value":
            branch_input = value_states
        else:
            residual = value_states - av
            # Shuffle across independent examples at the same position. This
            # preserves the residual distribution and strict LM causality; a
            # token-axis permutation could leak future-token information.
            branch_input = torch.roll(residual, shifts=1, dims=0)
        if self.residual_detach:
            branch_input = branch_input.detach()
        return branch_input

    def _residual_bshd(self, av: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        if self.source == "av":
            branch_input = av
        elif self.source == "value":
            branch_input = value_states
        else:
            branch_input = torch.roll(value_states - av, shifts=1, dims=0)
        if self.residual_detach:
            branch_input = branch_input.detach()
        return branch_input


def replace_lrcp_source(model, source: str) -> int:
    """Replace attached LRCP modules with exact-budget source controls."""

    replacements: list[tuple[object, LRCP]] = []
    for parent in model.modules():
        module = getattr(parent, "lrcp", None)
        if isinstance(module, LRCP) and not isinstance(module, LRCPSourceControl):
            replacements.append((parent, module))
    for parent, old in replacements:
        new = LRCPSourceControl(
            old.num_heads,
            old.head_dim,
            old.rank,
            old.gamma,
            down_init_std=0.0,
            up_init_std=0.0,
            residual_detach=old.residual_detach,
            telemetry_prefix=old.telemetry_prefix,
            source=source,
        ).to(device=old.down.device, dtype=old.down.dtype)
        with torch.no_grad():
            new.down.copy_(old.down)
            new.up.copy_(old.up)
        new.stats_enabled = old.stats_enabled
        parent.lrcp = new
    return len(replacements)
