from __future__ import annotations

# pyright: reportOptionalMemberAccess=false, reportPrivateUsage=false
import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import queue_a2d_r2k3_same_resolution_factorial as queue
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as uniform


def test_factorial_has_every_width_by_resolution_mask_once() -> None:
    assert len(runner.VARIANTS) == 48
    assert len(runner.TRAIN_VARIANTS) == 44
    assert len(runner.REFERENCE_VARIANTS) == 4
    for width in runner.WIDTHS:
        placements = {
            runner.SPECS[variant].resolutions
            for variant in runner.VARIANTS
            if runner.SPECS[variant].width == width
        }
        assert len(placements) == 16
        assert () in placements
        assert runner.RESOLUTIONS in placements


def test_gpu_queues_are_disjoint_complete_and_deliberately_balanced() -> None:
    gpu0 = queue.JOBS_BY_GPU[0]
    gpu1 = queue.JOBS_BY_GPU[1]
    assert len(gpu0) == 20
    assert len(gpu1) == 24
    assert set(gpu0).isdisjoint(gpu1)
    assert set(gpu0).union(gpu1) == set(runner.TRAIN_VARIANTS)
    assert gpu1[:5] == (
        runner.variant_name(128, runner.RESOLUTIONS),
        runner.variant_name(96, ()),
        runner.variant_name(96, runner.RESOLUTIONS),
        runner.variant_name(64, ()),
        runner.variant_name(64, runner.RESOLUTIONS),
    )


def test_parameter_capacity_depends_only_on_width_and_block_count() -> None:
    config = runtime.model_config()
    expected = {
        64: (408_868, 91_904),
        96: (837_028, 196_224),
        128: (1_414_692, 339_456),
    }
    for width, (base_parameters, block_parameters) in expected.items():
        for placement in ((), (56,), (28,), (14,), (7,), runner.RESOLUTIONS):
            variant = runner.variant_name(width, placement)
            model = runner._build(variant, config)
            actual = sum(parameter.numel() for parameter in model.parameters())
            assert actual == base_parameters + len(placement) * block_parameters
            assert model.descriptor_dim == 16 * width


def test_width_builds_do_not_mutate_capacity_registries() -> None:
    config = runtime.model_config()
    capacity_specs = dict(capacity.SPECS)
    try:
        narrow = uniform._build(uniform.VARIANTS[0], config)
        assert narrow.stage1.pole_input_projection.output_modes == 64
        assert capacity_specs == capacity.SPECS

        canonical = capacity._build(capacity.K128_P128, config)
        assert canonical.stage1.pole_input_projection.output_modes == 128
    finally:
        capacity.SPECS.clear()
        capacity.SPECS.update(capacity_specs)


def test_postfusion_ratio_resolves_exact_stage_widths() -> None:
    spec = capacity.CapacitySpec(
        excitation_modes=(128, 128, 160, 128),
        pole_modes=(128, 128, 160, 128),
        post_hidden_ratio=1.5,
    )

    assert spec.post_hidden_modes == (192, 192, 240, 192)


def test_capacity_builder_applies_ratio_to_each_transition_output_width() -> None:
    spec = capacity.CapacitySpec(
        excitation_modes=(128, 128, 160, 128),
        pole_modes=(128, 128, 160, 128),
        post_hidden_ratio=1.5,
    )
    model = capacity._build_spec(spec, runtime.model_config())

    assert tuple(
        getattr(model, name).augmented.post_hidden
        for name in runtime.STAGE_NAMES[:3]
    ) == (192, 240, 192)
    config = capacity._variant_config_for_spec("ratio-test", spec)
    assert config["backbone"]["transition"]["post_fusion"] == {
        "operator": "target-width residual WL PostFusion",
        "hidden_ratio": 1.5,
        "hidden_modes_by_excitation_stage": [192, 192, 240, 192],
    }
