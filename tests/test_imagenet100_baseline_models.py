from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from side_experiments.dense_vision import (
    evaluate_imagenet100_baseline_robustness as evaluator,
)

zoo = importlib.import_module("imagenet100_baseline_models")


class _Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3, 100)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(values.mean(dim=(-2, -1)))


@pytest.mark.parametrize("variant", zoo.VARIANTS)
def test_all_declared_variants_build_raw_classifier_topologies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variant: str,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def source_checkout(_variant: str, _source_root: Path | None) -> Path:
        return tmp_path

    def module(
        _checkout: Path,
        module_name: str,
        _prefixes: tuple[str, ...],
        **_kwargs: object,
    ) -> object:
        def constructor(**kwargs: object) -> nn.Module:
            calls.append((module_name, dict(kwargs)))
            return _Classifier()

        if module_name == "cvnets.models.classification.edgeformer":
            return SimpleNamespace(edgeformer=lambda options: constructor(options=options))
        if module_name == "models.moganet":
            return SimpleNamespace(moganet_xtiny=constructor)
        if module_name == "model.lib_emo.emo2":
            return SimpleNamespace(EMO2_1M_k5_hybrid=constructor, EMO2_2M_k5_hybrid=constructor)
        return SimpleNamespace(tinynext_t=constructor, tinynext_s=constructor)

    monkeypatch.setattr(zoo, "_source_checkout", source_checkout)
    monkeypatch.setattr(zoo, "_module", module)
    model = zoo._build(variant, zoo.LightweightZooConfig())

    assert isinstance(model, _Classifier)
    assert not any(key.startswith("model.") for key in model.state_dict())
    assert len(calls) == 1
    module_name, kwargs = calls[0]
    if variant.startswith("tinynext_"):
        assert module_name == "models.tinynext"
        assert kwargs == {"pretrained": False, "num_classes": 100, "distillation": False}
    elif variant.startswith("emov2_"):
        assert module_name == "model.lib_emo.emo2"
        assert kwargs == {"pretrained": False, "num_classes": 100}
    elif variant == "moganet_xt":
        assert module_name == "models.moganet"
        assert kwargs == {"pretrained": False, "num_classes": 100}
    else:
        options = kwargs["options"]
        assert getattr(options, "model.classification.n_classes") == 100
        assert getattr(options, "model.classification.edge.scale") == (
            "scale_xs" if variant == "parcnet_xs" else "scale_s"
        )


def test_checkout_validation_requires_the_pinned_clean_official_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="missing pinned source checkout"):
        zoo._source_checkout("tinynext_t", tmp_path)

    with pytest.raises(RuntimeError, match="missing pinned source checkout"):
        zoo._source_checkout("parcnet_xs", tmp_path)


def test_evaluator_extracts_the_primary_emov2_mapping_logits() -> None:
    primary = torch.randn(1, 100)
    assert evaluator._logits({"out": primary, "out_kd": primary + 1}) is primary


@pytest.mark.parametrize("variant", zoo.VARIANTS)
def test_actual_pinned_architecture_accepts_its_raw_state_dict_when_available(
    variant: str,
) -> None:
    root_value = os.environ.get("LNET_IMAGENET100_BASELINE_SMOKE_ROOT")
    if root_value is None:
        pytest.skip("set LNET_IMAGENET100_BASELINE_SMOKE_ROOT to run official-source smoke")
    config = zoo.LightweightZooConfig(source_root=Path(root_value))
    model = zoo._build(variant, config)
    replica = zoo._build(variant, config)
    replica.load_state_dict(model.state_dict(), strict=True)
    assert tuple(model.state_dict()) == tuple(replica.state_dict())
    inputs = torch.randn(1, 3, 224, 224)
    model.eval()
    replica.eval()
    with torch.inference_mode():
        expected = evaluator._logits(model(inputs))
        assert expected.shape == (1, 100)
        torch.testing.assert_close(evaluator._logits(replica(inputs)), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("variant", zoo.VARIANTS)
def test_actual_pinned_architecture_supports_cuda_bf16_evaluation(variant: str) -> None:
    root_value = os.environ.get("LNET_IMAGENET100_BASELINE_SMOKE_ROOT")
    if root_value is None:
        pytest.skip("set LNET_IMAGENET100_BASELINE_SMOKE_ROOT to run official-source smoke")
    config = zoo.LightweightZooConfig(source_root=Path(root_value))
    model = zoo._build(variant, config).eval().cuda()
    inputs = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = evaluator._logits(model(inputs))
    assert logits.shape == (1, 100)
    assert bool(logits.isfinite().all())
