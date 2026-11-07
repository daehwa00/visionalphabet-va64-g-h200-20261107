from __future__ import annotations

import builtins
import sys
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from side_experiments.dense_vision import analyze_r2k3_information_compression as compression
from side_experiments.dense_vision import (
    evaluate_imagenet100_baseline_robustness as baseline_robustness,
)
from side_experiments.dense_vision import evaluate_imagenet100_robustness as r2k3_robustness
from side_experiments.dense_vision import queue_coco_scratch_detection as coco_queue
from side_experiments.dense_vision import run_scratch_segmentation as segmentation

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


class _PathCollapse(nn.Module):
    def forward_packed(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return real.mean(-2, keepdim=True), imag.mean(-2, keepdim=True)


class _ScanBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.path_collapse = _PathCollapse()
        self.pole_scan = type("PoleScan", (), {"product_gain_normalization": "pointwise"})()

    def _scan_contract(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, ...], tuple[Tensor, ...]]:
        pole = tuple(torch.full((1, 1, 1, real.shape[-1]), 0.5) for _ in range(4))
        return (real, imag), pole, pole

    def _project_memory(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return real, imag

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return real + 1.0, imag + 1.0


class _ImageFolder:
    def __init__(self, _root: Path, *, transform: object) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> tuple[Tensor, int]:
        return self.transform(Image.new("RGB", (24, 16), color=(127, 127, 127))), 0  # type: ignore[operator]


class _Classifier(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        return torch.tensor([[1.0, 0.0]], device=images.device).expand(images.shape[0], -1)


@pytest.mark.parametrize("module", [r2k3_robustness, baseline_robustness])
def test_robustness_help_does_not_load_optional_training_dependencies(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def import_without_training_dependencies(
        name: str, *args: object, **kwargs: object,
    ) -> ModuleType:
        if name in {"wandb", "run_imagenet100_lightweight_zoo"}:
            raise AssertionError(f"--help must not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_training_dependencies)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit) as exited:
        module.main()
    assert exited.value.code == 0
    assert "--checkpoint" in capsys.readouterr().out


def test_information_compression_uses_current_scan_contract() -> None:
    collector = compression.Collector(token_samples=8)
    state = (torch.ones(1, 4, 4, 2), torch.zeros(1, 4, 4, 2))
    compression._analyze_block(_ScanBlock(), state, collector, "test")
    report = collector.report()
    assert "test/U_reader" in report["features"]
    assert "test" in report["pole_redundancy"]


@pytest.mark.parametrize("module", [r2k3_robustness, baseline_robustness])
def test_imagenet_robustness_evaluation_supports_cpu(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(module.datasets, "ImageFolder", _ImageFolder)
    accuracy = module.evaluate(  # type: ignore[attr-defined]
        _Classifier(),
        tmp_path,
        torch.device("cpu"),
        kind="clean",
        severity=0.0,
        size=32,
        batch_size=1,
        workers=0,
    )
    assert accuracy == 100.0


def test_coco_ready_requires_directories_and_annotation_files(tmp_path: Path) -> None:
    assert not coco_queue._ready(tmp_path)
    (tmp_path / "train2017").mkdir()
    (tmp_path / "val2017").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations/instances_train2017.json").write_text("{}", encoding="utf-8")
    (tmp_path / "annotations/instances_val2017.json").write_text("{}", encoding="utf-8")
    assert coco_queue._ready(tmp_path)
    (tmp_path / "train2017").rmdir()
    (tmp_path / "train2017").write_text("not a directory", encoding="utf-8")
    assert not coco_queue._ready(tmp_path)


def test_segmentation_evaluation_rejects_an_entirely_ignored_dataset() -> None:
    class _Segmenter(nn.Module):
        def forward(self, images: Tensor) -> Tensor:
            return torch.zeros(images.shape[0], 2, *images.shape[-2:])

    with pytest.raises(ValueError, match="no valid target pixels"):
        segmentation.evaluate(
            _Segmenter(),
            [(torch.zeros(1, 3, 4, 4), torch.full((1, 4, 4), 255, dtype=torch.int64))],
            torch.device("cpu"),
            classes=2,
        )
