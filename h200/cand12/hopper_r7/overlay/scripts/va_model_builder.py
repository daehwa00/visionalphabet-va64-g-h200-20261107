"""Standalone model and optimizer builder for VA-Ablation-10-dw-terminal-rmsnorm."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import orthogonal
from va_packed_backbone import (
    PackedVA128Backbone,
    RepeatedSameResolutionBackbone,
    SameResolutionFactorialBackbone,
    enable_packed_va128_,
)

from lnet.classifier_heads import AffineCorrelationClassifier
from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
    ComplexScanStage,
    ModalFusionHead,
)
from lnet.complex_scan_transitions import (
    AugmentedComplexTransition,
    S2DPostCFFNCarryMainTransition,
    S2DPostFusionCFFNTransition,
)
from lnet.depthwise_memory_reader import replace_readers_
from lnet.image_layers import (
    ModeScaledTwoConvStem,
    ResidualPreComplexMixer,
    StandardizedAffineModalHead,
)
from lnet.factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.gated_post_fusion import (
    GatedComplexPostFusion,
    GatedPoleExcitationS2DTransition,
    resized_gated_transition,
)
from lnet.phase_gated_cffn import PhaseGatedComplexFFN
from lnet.phase_gated_transition import (
    PathOnlyCollapse,
    PhaseGatedModeResidualPathCollapse,
)
from lnet.pole_excitation_transition import PoleExcitationS2DPostFusionTransition
from lnet.pole_initialization import install_pole_initialization
from lnet.same_resolution_depth import SameResolutionPoleScanBlock
from lnet.memory_backbone import MemoryBackbone
from lnet.memory_training import enable_memory_training_
from lnet.terminal_imaginary_correlation import complex_correlation_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable

    from lnet.complex_scan_types import ComplexField

STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
RESOLUTIONS = (56, 28, 14, 7)
READER_RANK = 2
KERNEL_SIZE = 3
PATH_HIDDEN = 4
LEGACY_PATH_HIDDEN = 8
POST_HIDDEN = 192
EXTRA_BLOCK_SEED_STRIDE = 1_000_003
DEFAULT_CHANNELS = (64, 96, 128, 256)
DEFAULT_POLES = (160, 160, 160, 128)
DEFAULT_DEPTH = (2, 2, 6, 2)
PARAMETERS = 2_389_796


@dataclass(frozen=True, slots=True)
class _FourStageSpec:
    modes: tuple[int, int, int, int]
    stem_width: int
    mode_cffn_widths: tuple[int, int, int]
    augmented_widths: tuple[int, int, int]
    post_ffn_widths: tuple[int, int, int]

    @property
    def descriptor_dim(self) -> int:
        return 4 * sum(self.modes)


@dataclass(frozen=True, slots=True)
class _StageSpec:
    excitation_modes: tuple[int, int, int, int]
    pole_modes: tuple[int, int, int, int]
    extra_blocks: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def descriptor_dim(self) -> int:
        return 4 * sum(self.pole_modes)

    @property
    def q4_dim(self) -> int:
        return 4 * self.pole_modes[-1]

    @property
    def post_hidden_modes(self) -> tuple[int, ...]:
        return tuple(round(modes * 2.0) for modes in self.excitation_modes)


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

    def __init__(self, backbone: nn.Module, *, normalize: bool = False) -> None:
        super().__init__(backbone)
        self.normalize = normalize

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

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        state = self.forward_features(inputs)
        terminal = self.backbone.terminal
        drive = terminal.pole_input_projection(state)
        px, py = terminal.pole_coefficients((*drive.shape[:-1], terminal.modes))
        descriptor = complex_correlation_descriptor(px, py, drive, normalize=self.normalize)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float(), terminal_only=True)


class _VariableFourStageBackbone(ComplexScanBackbone):
    """Join three adaptive transitions to a mode-matched terminal bank."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        stage3: ComplexScanStage,
        terminal: ComplexScanStage,
        descriptor_dim: int,
    ) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.stem = source.stem
        self.input_norm = source.input_norm
        self.precomplex_fc = source.precomplex_fc
        self.analysis = source.analysis
        self.stage1 = source.stage1
        self.stage2 = source.stage2
        self.stage3 = stage3
        self.terminal = terminal
        self.descriptor_dim = descriptor_dim
        self.classifier = source.classifier

    def complex_features(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        excitation = self._initial_excitation(inputs)
        state2, _ = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state3, _ = self.stage2(*state2)
        state3 = self._require_state(state3)
        state4, _ = self.stage3(*state3)
        return state2, state3, self._require_state(state4)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        excitation = self._initial_excitation(inputs)
        state2, descriptor1 = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state3, descriptor2 = self.stage2(*state2)
        state3 = self._require_state(state3)
        state4, descriptor3 = self.stage3(*state3)
        state4 = self._require_state(state4)
        _, descriptor4 = self.terminal(*state4)
        descriptor = torch.cat((descriptor1, descriptor2, descriptor3, descriptor4), dim=-1)
        if descriptor.shape[-1] != self.descriptor_dim:
            raise RuntimeError("four-stage descriptor contract changed")
        return descriptor


class _DeepModalFusionHead(ModalFusionHead):
    """Preserves seeded head initialization draws during Deep4 scaffold construction."""

    def __init__(self, source: ModalFusionHead, output_dim: int) -> None:
        nn.Module.__init__(self)
        self.input_dim = source.input_dim
        self.hidden_dim = source.hidden_dim
        self.second_hidden_dim = 256
        self.output_dim = output_dim
        self.standardizer = source.standardizer
        self.fusion = source.fusion
        self.activation = source.activation
        self.norm = source.norm
        self.refinement = nn.Linear(384, 256)
        self.refinement_activation = nn.GELU()
        self.refinement_norm = nn.RMSNorm(256)
        self.classifier = nn.Linear(256, output_dim)


def _install_four_stage_transition(stage: ComplexScanStage, post_hidden_modes: int) -> None:
    previous = stage.augmented
    if not isinstance(previous, AugmentedComplexTransition) or stage.carry_basis != "s2d":
        raise TypeError("seeded builder requires an augmented S2D transition")
    postcarry = S2DPostCFFNCarryMainTransition(
        modes=stage.modes,
        hidden_modes=previous.hidden_modes,
        output_modes=previous.output_modes,
        pole_paths=previous.input_modes // stage.modes,
        expansion=previous.ffn_input.output_modes // previous.hidden_modes,
        pole_scale_initial=1.0,
    )
    postcarry.copy_pole_branch_from(previous)
    postfusion = S2DPostFusionCFFNTransition(
        modes=stage.modes,
        hidden_modes=postcarry.hidden_modes,
        output_modes=postcarry.output_modes,
        pole_paths=postcarry.input_modes // stage.modes,
        expansion=postcarry.ffn_input.output_modes // postcarry.hidden_modes,
        pole_scale_initial=1.0,
        post_hidden_modes=post_hidden_modes,
        post_layer_scale_initial=0.1,
    )
    postfusion.copy_pole_branch_from(postcarry)
    if postfusion.carry_weight is not None and postcarry.carry_weight is not None:
        postfusion.carry_weight.data.copy_(postcarry.carry_weight.data)
    elif postfusion.carry_projection is not None and postcarry.carry_projection is not None:
        postfusion.carry_projection.load_state_dict(postcarry.carry_projection.state_dict())
    else:
        raise TypeError("PostCarry and PostFusion carry projections do not match")
    postfusion.pole_scale.data.copy_(postcarry.pole_scale.data)
    stage.augmented = postfusion


def _build_four_stage(spec: _FourStageSpec, config: ComplexScanConfig) -> ComplexScanBackbone:
    active = replace(
        config,
        stem_width=spec.stem_width,
        use_precomplex_fc=True,
        precomplex_fc_layers=2,
        modes=spec.modes[:3],
        augmented_widths=spec.augmented_widths[:2],
        carry_bases=("s2d", "s2d"),
        quadrant_path_mode_cffn_widths=spec.mode_cffn_widths[:2],
        quadrant_path_cffn_widths=(16, 16),
        quadratic_rank=64,
        fusion_width=384,
        dual_fusion_lrq_head=True,
    )
    source = ComplexScanBackbone(active)
    if not isinstance(source.precomplex_fc, nn.Sequential):
        raise TypeError("seeded builder is missing its pre-complex projection")
    layers = list(source.precomplex_fc.children())
    source.precomplex_fc = nn.Sequential(layers[0], layers[1], layers[2], nn.Identity())
    _install_four_stage_transition(source.stage1, spec.post_ffn_widths[0])
    _install_four_stage_transition(source.stage2, spec.post_ffn_widths[1])
    stage3 = ComplexScanStage(
        spec.modes[2],
        maximum_phase=math.pi * 0.60,
        output_modes=spec.modes[3],
        augmented_width=spec.augmented_widths[2],
        carry_basis=active.carry_bases[-1],
        carry_merge=active.carry_merge,
        carry_scale_initial=active.carry_scale_initial,
        coherence_gated_carry=active.coherence_gated_carry,
        quadrant_path_mode_cffn_width=spec.mode_cffn_widths[2],
        quadrant_path_cffn_width=16,
        stage_residual_scale_initial=active.stage_residual_scale_initial,
        scan_memory_policy=active.scan_memory_policy,
        damping_min=active.damping_min,
        damping_max=active.damping_max,
    )
    _install_four_stage_transition(stage3, spec.post_ffn_widths[2])
    terminal = ComplexScanStage(
        spec.modes[3],
        maximum_phase=math.pi * 0.65,
        output_modes=None,
        scan_memory_policy=active.scan_memory_policy,
        damping_min=active.damping_min,
        damping_max=active.damping_max,
    )
    model = _VariableFourStageBackbone(source, stage3, terminal, spec.descriptor_dim)
    affine = StandardizedAffineModalHead(spec.descriptor_dim, config.output_dim)
    fusion_source = ModalFusionHead(spec.descriptor_dim, 384, config.output_dim)
    fusion = _DeepModalFusionHead(fusion_source, config.output_dim)
    model.classifier = AffineCorrelationClassifier(
        spec.descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=1.0,
    )
    return cast("ComplexScanBackbone", model)


def _make_reader(excitation_modes: int, pole_modes: int) -> FactorizedComplexConv2dReader:
    reader = FactorizedComplexConv2dReader(
        excitation_modes,
        pole_modes,
        rank=READER_RANK,
        kernel_size=KERNEL_SIZE,
        match_input_rms=True,
    )
    if excitation_modes == pole_modes:
        reader.initialize_orthogonal_()
    else:
        reader.initialize_semi_orthogonal_()
    return reader


def _make_block(
    *,
    width: int,
    pole_modes: int,
    resolution: int,
    pole_template: ComplexScanStage,
    post_hidden: int,
    seed_offset: int = 0,
) -> SameResolutionPoleScanBlock:
    initial_seed = torch.initial_seed()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((initial_seed + resolution * 1_001 + seed_offset) % (2**63 - 1))
        return SameResolutionPoleScanBlock(
            width,
            pole_modes=pole_modes,
            reader_rank=READER_RANK,
            kernel_size=KERNEL_SIZE,
            pole_template=pole_template,
            post_hidden=post_hidden,
        )


def _build_stage_allocation(spec: _StageSpec, config: ComplexScanConfig) -> ComplexScanBackbone:
    poles = spec.pole_modes
    deep4 = _FourStageSpec(
        modes=poles,
        stem_width=2 * max(spec.excitation_modes[0], poles[0]),
        mode_cffn_widths=tuple(2 * modes for modes in poles[:3]),
        augmented_widths=tuple(2 * modes for modes in poles[:3]),
        post_ffn_widths=tuple(2 * modes for modes in poles[:3]),
    )
    model = _build_four_stage(deep4, config)
    excitation_width = 2 * spec.excitation_modes[0]
    model.stem = ModeScaledTwoConvStem(spec.excitation_modes[0], config.stem_strides, hidden_width=32)
    if model.config.stem_width == excitation_width:
        model.precomplex_fc = ResidualPreComplexMixer(cast("nn.Module", model.precomplex_fc))
    else:
        model.input_norm = nn.RMSNorm(excitation_width)
        model.precomplex_fc = ResidualPreComplexMixer(
            nn.Sequential(
                nn.Linear(excitation_width, excitation_width),
                nn.GELU(),
                nn.Linear(excitation_width, excitation_width),
                nn.Identity(),
            )
        )
    analysis = nn.Linear(excitation_width, excitation_width, bias=False)
    nn.init.orthogonal_(analysis.weight)
    orthogonal(analysis, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
    model.analysis = analysis

    for index, name in enumerate(STAGE_NAMES[:3]):
        stage = cast("ComplexScanStage", getattr(model, name))
        excitation_modes = spec.excitation_modes[index]
        pole_modes = spec.pole_modes[index]
        output_modes = spec.excitation_modes[index + 1]
        stage.input_modes = excitation_modes
        stage.output_modes = output_modes
        stage.carry_basis = "s2d"
        stage.transition = None
        stage.interaction = None
        stage.widely_bridge = None
        stage.bridge = None
        stage.output_norm = None
        stage.post_transition_ffn = None
        stage.quadrant_path_mode_combiner = PathOnlyCollapse(
            PhaseGatedModeResidualPathCollapse(
                pole_modes, mode_hidden=pole_modes, path_hidden=LEGACY_PATH_HIDDEN
            )
        )
        stage.augmented = PoleExcitationS2DPostFusionTransition(
            pole_modes,
            excitation_modes,
            output_modes,
            post_hidden=spec.post_hidden_modes[index + 1],
        )
        with torch.random.fork_rng(devices=[]):
            stage.pole_input_projection = _make_reader(excitation_modes, pole_modes)

    terminal = model.terminal
    terminal.input_modes = spec.excitation_modes[-1]
    with torch.random.fork_rng(devices=[]):
        terminal.pole_input_projection = _make_reader(spec.excitation_modes[-1], spec.pole_modes[-1])

    install_pole_initialization(model, STAGE_NAMES)
    model.classifier = TerminalAffineClassifier(
        spec.descriptor_dim, spec.q4_dim, config.output_dim
    )

    for name in STAGE_NAMES[:3]:
        stage = cast("ComplexScanStage", getattr(model, name))
        source_reader = cast("FactorizedComplexConv2dReader", stage.pole_input_projection)
        pre_norm_reader = FactorizedComplexConv2dReader(
            source_reader.input_modes,
            source_reader.output_modes,
            rank=source_reader.rank,
            kernel_size=source_reader.kernel_size,
            variance_epsilon=source_reader.variance_epsilon,
            normalize_input=True,
            match_input_rms=False,
        )
        pre_norm_reader.load_state_dict(source_reader.state_dict(), strict=False)
        stage.pole_input_projection = pre_norm_reader
        source_transition = cast("PoleExcitationS2DPostFusionTransition", stage.augmented)
        stage.augmented = GatedPoleExcitationS2DTransition(
            source_transition.input_modes,
            source_transition.excitation_modes,
            source_transition.output_modes,
            post_hidden=source_transition.post_hidden,
        )

    for name in STAGE_NAMES[:3]:
        stage = cast("ComplexScanStage", getattr(model, name))
        gated = cast("GatedPoleExcitationS2DTransition", stage.augmented)
        stage.augmented = resized_gated_transition(gated, POST_HIDDEN)

    templates = {
        56: model.stage1,
        28: model.stage2,
        14: cast("Any", model).stage3,
        7: model.terminal,
    }
    blocks = {
        resolution: _make_block(
            width=spec.excitation_modes[index],
            pole_modes=spec.pole_modes[index],
            resolution=resolution,
            pole_template=templates[resolution],
            post_hidden=POST_HIDDEN,
        )
        for index, resolution in enumerate(RESOLUTIONS)
    }
    factorial_model = SameResolutionFactorialBackbone(model, blocks)
    for name in STAGE_NAMES[:3]:
        stage = getattr(factorial_model, name)
        stage.quadrant_path_mode_combiner = PathOnlyCollapse(stage.modes, path_hidden=PATH_HIDDEN)
    for block in factorial_model.same_resolution_blocks.values():
        block.path_collapse = PathOnlyCollapse(block.pole_modes, path_hidden=PATH_HIDDEN)

    templates = {
        56: factorial_model.stage1,
        28: factorial_model.stage2,
        14: cast("Any", factorial_model).stage3,
        7: factorial_model.terminal,
    }
    extras: dict[int, tuple[SameResolutionPoleScanBlock, ...]] = {}
    for stage_index, (resolution, count) in enumerate(
        zip(RESOLUTIONS, spec.extra_blocks, strict=True)
    ):
        if count <= 0:
            continue
        stage_blocks = []
        for repeat_index in range(count):
            block = _make_block(
                width=spec.excitation_modes[stage_index],
                pole_modes=spec.pole_modes[stage_index],
                resolution=resolution,
                pole_template=templates[resolution],
                post_hidden=POST_HIDDEN,
                seed_offset=EXTRA_BLOCK_SEED_STRIDE * (repeat_index + 1),
            )
            block.path_collapse = PathOnlyCollapse(
                spec.pole_modes[stage_index], path_hidden=PATH_HIDDEN
            )
            stage_blocks.append(block)
        extras[resolution] = tuple(stage_blocks)
    return cast("ComplexScanBackbone", RepeatedSameResolutionBackbone(factorial_model, extras))


@torch.no_grad()
def _pair_common_initialization_(
    target: ComplexScanBackbone,
    reference: ComplexScanBackbone,
) -> None:
    source = reference.state_dict()
    for name, value in target.state_dict().items():
        source_value = source.get(name)
        if source_value is not None and source_value.shape == value.shape:
            value.copy_(source_value)


def model_config(*, output_dim: int = 100) -> ComplexScanConfig:
    return ComplexScanConfig(output_dim=output_dim, stem_strides=(2, 2))


def build_dw_terminal_rmsnorm(
    config: ComplexScanConfig | None = None,
    *,
    channels: tuple[int, int, int, int] = DEFAULT_CHANNELS,
    poles: tuple[int, int, int, int] = DEFAULT_POLES,
    depth: tuple[int, int, int, int] = DEFAULT_DEPTH,
    normalize: bool = False,
) -> VisionAlphabetIN100:
    """Build VA-Ablation-10-dw-terminal-rmsnorm preserving exact seed-501 initialization."""
    active_config = config if config is not None else model_config()
    if active_config.output_dim != 100:
        raise ValueError("VA-Ablation-10-dw-terminal-rmsnorm requires output_dim=100")
    seed = torch.initial_seed()
    spec = _StageSpec(
        excitation_modes=cast("tuple[int, int, int, int]", tuple(c // 2 for c in channels)),
        pole_modes=poles,
        extra_blocks=cast("tuple[int, int, int, int]", tuple(d - 2 for d in depth)),
    )
    model = _build_stage_allocation(spec, active_config)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, GatedComplexPostFusion):
                hidden = round(1.5 * child.modes)
                setattr(parent, name, GatedComplexPostFusion(child.modes, hidden))
                if isinstance(parent, GatedPoleExcitationS2DTransition):
                    parent.post_hidden = hidden
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = _build_stage_allocation(
            _StageSpec((128, 128, 128, 128), (160, 160, 160, 128), extra_blocks=(0, 0, 4, 0)),
            active_config,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            base_ref = _build_stage_allocation(
                _StageSpec((128, 128, 128, 128), (160, 160, 160, 160), extra_blocks=(0, 0, 4, 0)),
                active_config,
            )
        _pair_common_initialization_(reference, base_ref)
    _pair_common_initialization_(model, reference)
    for module in model.modules():
        if isinstance(module, FactorizedComplexConv2dReader):
            module.execution_mode = "factorized_triton"
    enable_packed_va128_(model)
    result = VisionAlphabetIN100(model, normalize=normalize)

    old = result.classifier
    with torch.random.fork_rng(devices=[]):
        new = TerminalAffineClassifier(1024, 1024, 100)
    with torch.no_grad():
        new.affine.linear.weight.zero_()
        new.affine.linear.weight[:, :512].copy_(old.affine.linear.weight)
        new.affine.linear.bias.copy_(old.affine.linear.bias)
        new.affine.standardizer.running_mean[:512].copy_(old.affine.standardizer.running_mean)
        new.affine.standardizer.running_var[:512].copy_(old.affine.standardizer.running_var)
        new.affine.standardizer.num_batches_tracked.copy_(old.affine.standardizer.num_batches_tracked)
    result.backbone.classifier = new
    result.backbone.descriptor_dim = 1024
    enable_memory_training_(result)
    replace_readers_(result)
    terminal_reader = result.backbone.terminal.pole_input_projection
    stage4_norm = result.backbone.same_resolution_blocks["7"].reader.input_norm
    if stage4_norm is None:
        raise RuntimeError("expected non-null input_norm on stage4 same-resolution reader")
    terminal_reader.input_norm = copy.deepcopy(stage4_norm)
    terminal_reader.match_input_rms = False
    if sum(p.numel() for p in result.parameters()) != PARAMETERS:
        raise RuntimeError(f"parameter count mismatch: expected {PARAMETERS}")
    return result


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
    destination = os.environ.get("VA_IN100_PROGRESS_PATH")
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
