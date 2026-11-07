"""Pin the R2K3 training recipe to the frozen legacy ancestry modules.

``a2d_r2k3_runtime.run`` used to reach its collaborators by walking module
aliases through the experiment-inheritance chain.  Those walks are now spelled
out as module names.  These tests prove the names resolve to exactly the same
modules and the same function objects, so the rewrite cannot silently rebind
the training recipe.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from scripts import a2d_r2k3_runtime as runtime


def _legacy():
    return runtime._legacy_module(runtime._LEGACY_TRAINING_MODULE)


def test_named_modules_match_the_original_alias_walk() -> None:
    legacy = _legacy()
    source = legacy._ramp().canonical8.fair_init.backbone.deep4.baseline.baseline

    assert runtime._legacy_module(runtime._HARNESS_MODULE) is source.heads.harness
    assert runtime._legacy_module(runtime._HEADS_MODULE) is source.heads
    assert runtime._legacy_module(runtime._STRUCTURED_MODULE) is source.structured
    assert runtime._legacy_module(runtime._PREPARE_MODULE) is source


def test_metrics_module_matches_the_original_alias_walk() -> None:
    walked = _legacy().control.no_pg.control.base.local_reader.control.control
    assert runtime._legacy_module(runtime._METRICS_MODULE) is walked


def test_recipe_callables_are_the_frozen_objects() -> None:
    legacy = _legacy()
    source = legacy._ramp().canonical8.fair_init.backbone.deep4.baseline.baseline
    metrics = legacy.control.no_pg.control.base.local_reader.control.control

    harness = runtime._legacy_module(runtime._HARNESS_MODULE)
    heads = runtime._legacy_module(runtime._HEADS_MODULE)
    structured = runtime._legacy_module(runtime._STRUCTURED_MODULE)
    prepare = runtime._legacy_module(runtime._PREPARE_MODULE)

    assert harness.main is source.heads.harness.main
    assert harness.runner_bindings is source.heads.harness.runner_bindings
    assert prepare._prepare_model is source._prepare_model
    assert structured._train_epoch is source.structured._train_epoch
    assert heads._evaluate is source.heads._evaluate
    assert heads._summarize is source.heads._summarize
    assert heads._training_objective is source.heads._training_objective
    assert heads._after_training_batch is source.heads._after_training_batch
    assert (
        runtime._legacy_module(runtime._METRICS_MODULE)._wandb_model_metrics
        is metrics._wandb_model_metrics
    )


def test_legacy_module_cache_returns_one_instance_per_name() -> None:
    first = runtime._legacy_module(runtime._HEADS_MODULE)
    second = runtime._legacy_module(runtime._HEADS_MODULE)
    assert first is second
