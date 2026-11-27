"""Standalone model and optimizer builder for VA-Ablation-10-dw-terminal-rmsnorm."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from alphabet_packed_backbone import (
    PackedVA128Backbone,
    RepeatedSameResolutionBackbone,
    SameResolutionFactorialBackbone,
    enable_packed_va128_,
)
from torch import Tensor, nn
from torch.nn.utils.parametrizations import orthogonal

from alphabet.classifier_heads import AffineCorrelationClassifier
from alphabet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage
from alphabet.depthwise_memory_reader import replace_readers_
from alphabet.factorized_complex_scan_reader import FactorizedComplexConv2dReader
from alphabet.gated_post_fusion import GatedPoleExcitationS2DTransition
from alphabet.image_layers import (
    ModeScaledTwoConvStem,
    ResidualPreComplexMixer,
    StandardizedAffineModalHead,
)
from alphabet.memory_backbone import ChannelRMSNorm, FactorizedMemoryReader, MemoryBackbone
from alphabet.memory_training import enable_memory_training_
from alphabet.phase_gated_cffn import PhaseGatedComplexFFN
from alphabet.phase_gated_transition import PathOnlyCollapse
from alphabet.pole_initialization import install_pole_initialization
from alphabet.same_resolution_depth import SameResolutionPoleScanBlock
from alphabet.terminal_imaginary_correlation import complex_correlation_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable


STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
RESOLUTIONS = (56, 28, 14, 7)
READER_RANK = 2
KERNEL_SIZE = 3
PATH_HIDDEN = 4
POST_HIDDEN_RATIO = 1.5
DEFAULT_CHANNELS = (64, 96, 128, 256)
DEFAULT_POLES = (160, 160, 160, 128)
DEFAULT_DEPTH = (2, 2, 6, 2)
PARAMETERS = 2_389_796


class TerminalAffineClassifier(AffineCorrelationClassifier):
    """Select terminal raw-Q coordinates for a dynamic pole schedule."""

    def __init__(self, descriptor_dim: int, q4_dim: int, output_dim: int) -> None:
        super().__init__(
            q4_dim,
            output_dim,
            main="affine",
            affine=StandardizedAffineModalHead(q4_dim, output_dim),
            fusion=None,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.0,
        )
        self.full_descriptor_dim = descriptor_dim
        self.q4_dim = q4_dim

    def select_q4(self, descriptor: Tensor) -> Tensor:
        if descriptor.shape[-1] != self.full_descriptor_dim:
            raise ValueError("capacity Q4 head received an incompatible full descriptor")
        return descriptor[..., -self.q4_dim :]

    def forward(
        self,
        descriptor: Tensor,
        *,
        terminal_only: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.forward_q4(descriptor if terminal_only else self.select_q4(descriptor))

    def forward_q4(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if descriptor.shape[-1] != self.q4_dim:
            raise ValueError("capacity Q4 head received an incompatible terminal descriptor")
        return super().forward(descriptor)


class VisionAlphabetIN100(MemoryBackbone):
    """Expose the affine head and compiled training hook on top of PackedVA128Backbone."""

    architecture = "va-terminal-rawg-dw-terminal-rmsnorm-v1"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        normalize: bool = False,
        forward_vertical_only: bool = False,
        non_redundant_four_directions: bool = False,
    ) -> None:
        super().__init__(backbone)
        self.normalize = normalize
        self.forward_vertical_only = forward_vertical_only
        self.non_redundant_four_directions = non_redundant_four_directions

    @property
    def classifier(self) -> TerminalAffineClassifier:
        classifier = self.backbone.classifier
        if not isinstance(classifier, TerminalAffineClassifier):
            raise TypeError(
                "ImageNet-100 real-memory training requires the original Q4 affine head"
            )
        return classifier

    def prepare_for_compiled_training_(self) -> VisionAlphabetIN100:
        cast("PackedVA128Backbone", self.backbone).prepare_for_compiled_training_()
        return self

    def terminal_descriptor(
        self, terminal: Any, state: Tensor, *, forward_vertical_only: bool | None = None
    ) -> Tensor:
        drive = terminal.pole_input_projection(state)
        px, py = terminal.pole_coefficients((*drive.shape[:-1], terminal.modes))
        return complex_correlation_descriptor(
            px, py, drive, normalize=self.normalize,
            forward_vertical_only=(
                self.forward_vertical_only if forward_vertical_only is None else forward_vertical_only
            ),
            non_redundant_four_directions=(
                self.non_redundant_four_directions if forward_vertical_only is None else False
            ),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        state = self.forward_features(inputs)
        descriptor = self.terminal_descriptor(self.backbone.terminal, state)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float(), terminal_only=True)


class MultiScaleVisionAlphabetIN100(VisionAlphabetIN100):
    """Classify paired complex correlations from the 14x14 and 7x7 features."""

    architecture = "va-terminal-multiscale-s3s4-live-rms-v2"

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state_s3, state_s4 = self.forward_feature_scales(inputs)
        desc_s3 = self.terminal_descriptor(self.backbone.terminal_s3, state_s3, forward_vertical_only=True)
        desc_s4 = self.terminal_descriptor(self.backbone.terminal, state_s4, forward_vertical_only=False)
        re_s3, im_s3 = desc_s3.chunk(2, dim=-1)
        re_s4, im_s4 = desc_s4.chunk(2, dim=-1)
        descriptor = torch.cat((re_s3, re_s4, im_s3, im_s4), dim=-1)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float(), terminal_only=True)


class _FourStageBackbone(ComplexScanBackbone):
    """Construct the stem, pole banks and terminal head used by the packed backbone."""

    def __init__(
        self,
        config: ComplexScanConfig,
        excitation_modes: tuple[int, ...],
        poles: tuple[int, int, int, int],
        post_hidden_ratio: float,
    ) -> None:
        nn.Module.__init__(self)
        width = 2 * excitation_modes[0]
        self.config = replace(
            config,
            stem_width=2 * max(excitation_modes[0], poles[0]),
            use_precomplex_fc=True,
            precomplex_fc_layers=2,
            modes=poles[:3],
            augmented_widths=tuple(2 * p for p in poles[:2]),
            carry_bases=("s2d", "s2d"),
            quadrant_path_mode_cffn_widths=tuple(2 * p for p in poles[:2]),
            quadrant_path_cffn_widths=(16, 16),
            quadratic_rank=64,
            fusion_width=384,
            dual_fusion_lrq_head=True,
        )
        self.config.validate()
        self.stem = ModeScaledTwoConvStem(excitation_modes[0], config.stem_strides, hidden_width=32)
        self.input_norm = (
            nn.RMSNorm(width)
            if self.config.stem_width != width or config.use_post_stem_rmsnorm
            else nn.Identity()
        )
        self.precomplex_fc = ResidualPreComplexMixer(
            nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.Identity()
            )
        )
        self.analysis = nn.Linear(width, width, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        orthogonal(self.analysis, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
        for index, (name, phase, pole_modes, input_modes) in enumerate(
            zip(STAGE_NAMES, (0.75, 0.70, 0.60, 0.65), poles, excitation_modes, strict=True)
        ):
            stage = ComplexScanStage(
                pole_modes,
                maximum_phase=math.pi * phase,
                output_modes=None,
                scan_memory_policy=config.scan_memory_policy,
                damping_min=config.damping_min,
                damping_max=config.damping_max,
            )
            stage.input_modes = input_modes
            if index < 3:
                output_modes = excitation_modes[index + 1]
                stage.output_modes = output_modes
                stage.carry_basis = "s2d"
                stage.augmented = GatedPoleExcitationS2DTransition(
                    pole_modes,
                    input_modes,
                    output_modes,
                    post_hidden=round(post_hidden_ratio * output_modes),
                )
                stage.quadrant_path_mode_combiner = PathOnlyCollapse(
                    pole_modes, path_hidden=PATH_HIDDEN
                )
            stage.pole_input_projection = _make_reader(
                input_modes,
                pole_modes,
                normalize_input=index < 3,
            )
            setattr(self, name, stage)
        install_pole_initialization(self, STAGE_NAMES)
        self.descriptor_dim = 8 * poles[-1]
        self.classifier = TerminalAffineClassifier(
            self.descriptor_dim, self.descriptor_dim, config.output_dim
        )


def _make_reader(
    excitation_modes: int,
    pole_modes: int,
    *,
    normalize_input: bool,
) -> FactorizedComplexConv2dReader:
    reader = FactorizedComplexConv2dReader(
        excitation_modes,
        pole_modes,
        rank=READER_RANK,
        kernel_size=KERNEL_SIZE,
        normalize_input=normalize_input,
        match_input_rms=not normalize_input,
    )
    if excitation_modes == pole_modes:
        reader.initialize_orthogonal_()
    else:
        reader.initialize_semi_orthogonal_()
    return reader


def _build_backbone(
    config: ComplexScanConfig,
    excitation_modes: tuple[int, ...],
    poles: tuple[int, int, int, int],
    depth: tuple[int, int, int, int],
    post_hidden_ratio: float,
) -> PackedVA128Backbone:
    source = _FourStageBackbone(config, excitation_modes, poles, post_hidden_ratio)
    blocks = {}
    extras = {}
    for name, resolution, width, pole_modes, count in zip(
        STAGE_NAMES,
        RESOLUTIONS,
        excitation_modes,
        poles,
        depth,
        strict=True,
    ):
        stage = getattr(source, name)
        stage_blocks = []
        for _ in range(max(1, count - 1)):
            block = SameResolutionPoleScanBlock(
                width,
                pole_modes=pole_modes,
                reader_rank=READER_RANK,
                kernel_size=KERNEL_SIZE,
                pole_template=stage,
                post_hidden=round(post_hidden_ratio * width),
            )
            block.path_collapse = PathOnlyCollapse(pole_modes, path_hidden=PATH_HIDDEN)
            stage_blocks.append(block)
        blocks[resolution] = stage_blocks[0]
        if len(stage_blocks) > 1:
            extras[resolution] = tuple(stage_blocks[1:])
    scaffold = SameResolutionFactorialBackbone(source, blocks)
    backbone = RepeatedSameResolutionBackbone(scaffold, extras)
    return cast("PackedVA128Backbone", enable_packed_va128_(backbone))


def model_config(*, output_dim: int = 100) -> ComplexScanConfig:
    return ComplexScanConfig(output_dim=output_dim, stem_strides=(2, 2))


def build_dw_terminal_rmsnorm(
    config: ComplexScanConfig | None = None,
    *,
    channels: tuple[int, int, int, int] = DEFAULT_CHANNELS,
    poles: tuple[int, int, int, int] = DEFAULT_POLES,
    depth: tuple[int, int, int, int] = DEFAULT_DEPTH,
    post_hidden_ratio: float = POST_HIDDEN_RATIO,
    normalize: bool = False,
    factorized_reader: bool = False,
    forward_vertical_only: bool = False,
    non_redundant_four_directions: bool = False,
) -> VisionAlphabetIN100:
    """Build VA-Ablation-10-dw-terminal-rmsnorm with a terminal affine head."""
    active_config = config if config is not None else model_config()
    backbone = _build_backbone(active_config, tuple(c // 2 for c in channels), poles, depth, post_hidden_ratio)
    for module in backbone.modules():
        if isinstance(module, FactorizedComplexConv2dReader):
            module.execution_mode = "factorized_triton"
    result = VisionAlphabetIN100(
        backbone, normalize=normalize, forward_vertical_only=forward_vertical_only,
        non_redundant_four_directions=non_redundant_four_directions,
    )
    clean = forward_vertical_only and not non_redundant_four_directions
    if clean:
        descriptor_dim = backbone.descriptor_dim // 2
        old = result.classifier.affine.linear
        with torch.random.fork_rng(devices=[]):
            backbone.classifier = TerminalAffineClassifier(
                descriptor_dim, descriptor_dim, active_config.output_dim
            )
        with torch.no_grad():
            new = result.classifier.affine.linear
            new.weight.zero_()
            new.weight[:, :descriptor_dim // 2].copy_(old.weight[:, :descriptor_dim // 2])
            new.bias.copy_(old.bias)
        backbone.descriptor_dim = descriptor_dim
    enable_memory_training_(result)
    if not factorized_reader:
        replace_readers_(result)
    terminal_reader = result.backbone.terminal.pole_input_projection
    stage4_norm = result.backbone.same_resolution_blocks["7"].reader.input_norm
    if stage4_norm is None:
        raise RuntimeError("expected non-null input_norm on stage4 same-resolution reader")
    terminal_reader.input_norm = copy.deepcopy(stage4_norm)
    terminal_reader.match_input_rms = False
    expected_parameters = 2_988_260 if factorized_reader else PARAMETERS
    if clean:
        expected_parameters -= 51_200
    if (
        (channels, poles, depth) == (DEFAULT_CHANNELS, DEFAULT_POLES, DEFAULT_DEPTH)
        and active_config.output_dim == 100
        and sum(p.numel() for p in result.parameters()) != expected_parameters
    ):
        raise RuntimeError(f"parameter count mismatch: expected {expected_parameters}")
    return result


def add_multiscale_readout(model: VisionAlphabetIN100) -> MultiScaleVisionAlphabetIN100:
    backbone = model.backbone
    terminal = backbone.terminal
    terminal_projection = terminal.pole_input_projection
    with torch.random.fork_rng(devices=[]):
        terminal_s3 = copy.deepcopy(terminal)
        terminal_s3.input_modes = 64
        reader = _make_reader(64, terminal_s3.modes, normalize_input=False)
        reader.execution_mode = terminal_projection.execution_mode
        reader.match_input_rms = False
        projection = FactorizedMemoryReader(reader)
        norm = terminal_projection.input_norm
        projection.input_norm = ChannelRMSNorm(128, norm.epsilon)
        projection._real_training_autotune = terminal_projection._real_training_autotune
        projection._real_training_vector_gradient = (
            terminal_projection._real_training_vector_gradient
        )
        terminal_s3.pole_input_projection = projection.to(terminal_projection.point_weight)
        terminal_s3.train(terminal.training)
        output_dim = model.classifier.affine.linear.out_features
        classifier = TerminalAffineClassifier(1536, 1536, output_dim)
        classifier.to(model.classifier.affine.linear.weight).train(model.classifier.training)
    with torch.no_grad():
        # Both imaginary planes start with zero classifier weights, as in A2.
        classifier.affine.linear.weight[:, 768:].zero_()
        classifier.affine.linear.weight[:, 256:768].copy_(model.classifier.affine.linear.weight[:, :512])
        classifier.affine.linear.bias.copy_(model.classifier.affine.linear.bias)
    backbone.terminal_s3 = terminal_s3
    backbone.classifier = classifier
    backbone.descriptor_dim = 1536
    # Both banks have 128 poles. Copying the initialized terminal preserves its exact atlas.
    model.__class__ = MultiScaleVisionAlphabetIN100
    return cast("MultiScaleVisionAlphabetIN100", model)


def build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
    *,
    variant: str = "VA-Ablation-10-dw-terminal-rmsnorm",
    seed: int = 501,
    atomic_json: Callable[[Path, object], None] | None = None,
) -> torch.optim.Optimizer:
    """Build the pole-aware AdamW optimizer matching VA-Ablation-10-dw-terminal-rmsnorm."""
    phase_gated_projection_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, PhaseGatedComplexFFN)
        for projection in (module.input_projection, module.output_projection)
        for parameter in projection.parameters()
    }
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    modal_no_decay: list[nn.Parameter] = []
    geometry: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter_name = name.rsplit(".", maxsplit=1)[-1]
        if id(parameter) in phase_gated_projection_ids:
            no_decay.append(parameter)
        elif "damping_logits" in parameter_name or parameter_name.startswith("phase_"):
            geometry.append(parameter)
        elif (
            "analysis" in name
            or "widely_bridge" in name
            or "augmented.direction_mixer" in name
            or "augmented.output_projection" in name
        ):
            modal_no_decay.append(parameter)
        elif parameter.ndim < 2 or "norm" in name or "initial_" in name or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = float(recipe["learning_rate"])
    groups = [
        {"params": decay, "lr": learning_rate, "weight_decay": recipe["weight_decay"]},
        {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
        {
            "params": modal_no_decay,
            "lr": learning_rate * recipe["modal_learning_rate_multiplier"],
            "weight_decay": 0.0,
        },
        {
            "params": geometry,
            "lr": learning_rate * recipe["pole_geometry_learning_rate_multiplier"],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, fused=bool(recipe.get("fused_optimizer", False)))
    destination = os.environ.get("ALPHABET_IMAGENET100_PROGRESS_PATH")
    if destination and atomic_json is not None:
        steps = 0

        def record(opt: torch.optim.Optimizer, _args: Any, _kwargs: Any) -> None:
            nonlocal steps
            steps += 1
            if steps != 1 and steps % 100:
                return
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(grads) and bool(torch.stack([g.isfinite().all() for g in grads]).all())
            atomic_json(
                Path(destination),
                {
                    "variant": variant,
                    "seed": seed,
                    "session_optimizer_steps": steps,
                    "finite_gradients": finite,
                    "current_learning_rates": [g["lr"] for g in opt.param_groups],
                    "unix_time": time.time(),
                },
            )
            if not finite:
                raise RuntimeError("nonfinite gradients")

        optimizer.register_step_post_hook(record)
    return optimizer
