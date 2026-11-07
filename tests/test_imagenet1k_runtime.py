from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from scripts import imagenet1k_runtime as runtime

if TYPE_CHECKING:
    from pathlib import Path


class TinyClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(4, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class IdentityMixup:
    def __init__(self, classes: int) -> None:
        self.classes = classes

    def __call__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return inputs, functional.one_hot(targets, self.classes).float()


def _task(root: Path, *, resume: bool = False, classes: int = 3) -> runtime.BaselineTask:
    return runtime.BaselineTask(
        phase="full",
        model_key="tiny",
        seed=501,
        learning_rate=1.0e-3,
        epochs=2,
        data_root=root / "data",
        output_dir=root,
        result_path=root / "results" / "result.json",
        checkpoint_path=root / "checkpoints" / "checkpoint.pt",
        batch_size=128,
        workers=0,
        num_classes=classes,
        resume=resume,
        source_sha256={"runner": "a" * 64},
    )


def _loader(task: runtime.BaselineTask, _device: torch.device) -> runtime.LoaderBundle:
    generator = torch.Generator().manual_seed(73)
    train = TensorDataset(
        torch.randn(8, 3, 4, 4, generator=generator),
        torch.arange(8) % task.num_classes,
    )
    validation = TensorDataset(
        torch.randn(4, 3, 4, 4, generator=generator),
        torch.arange(4) % task.num_classes,
    )
    train_generator = torch.Generator().manual_seed(task.seed)
    validation_generator = torch.Generator().manual_seed(task.seed + 1)
    return runtime.LoaderBundle(
        DataLoader(train, batch_size=task.batch_size, shuffle=True, generator=train_generator),
        DataLoader(validation, batch_size=task.batch_size, generator=validation_generator),
        train_generator,
        validation_generator,
    )


def _builder(_key: str, _source: str | Path | None, classes: int) -> nn.Module:
    return TinyClassifier(classes)


def test_cpu_checkpoint_resume_preserves_model_and_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uninterrupted = runtime.run_task(
        _task(tmp_path / "uninterrupted"),
        device=torch.device("cpu"),
        model_builder=_builder,
        loader_builder=_loader,
        mixup_factory=lambda task: IdentityMixup(task.num_classes),
    )
    interrupted_task = _task(tmp_path / "interrupted")
    real_emit = runtime._emit_progress
    interrupted = False

    def stop_after_checkpoint(_task: runtime.BaselineTask, payload: dict[str, object]) -> None:
        nonlocal interrupted
        if payload["epoch"] == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "_emit_progress", stop_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        runtime.run_task(
            interrupted_task,
            device=torch.device("cpu"),
            model_builder=_builder,
            loader_builder=_loader,
            mixup_factory=lambda task: IdentityMixup(task.num_classes),
        )
    monkeypatch.setattr(runtime, "_emit_progress", real_emit)
    resumed = runtime.run_task(
        replace(interrupted_task, resume=True),
        device=torch.device("cpu"),
        model_builder=_builder,
        loader_builder=_loader,
        mixup_factory=lambda task: IdentityMixup(task.num_classes),
    )

    resumed_history = [
        {key: value for key, value in row.items() if key != "training_seconds"}
        for row in resumed["history"]
    ]
    uninterrupted_history = [
        {key: value for key, value in row.items() if key != "training_seconds"}
        for row in uninterrupted["history"]
    ]
    assert resumed_history == uninterrupted_history
    left = torch.load(interrupted_task.checkpoint_path, map_location="cpu", weights_only=True)
    right = torch.load(
        _task(tmp_path / "uninterrupted").checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    for name, value in left["model"].items():
        assert torch.equal(value, right["model"][name])


@pytest.mark.parametrize("workers", [0, 1])
def test_default_timm_recipe_trains_on_a_tiny_imagefolder(tmp_path: Path, workers: int) -> None:
    for split in ("train", "val"):
        for class_name, color in (("zero", (255, 0, 0)), ("one", (0, 255, 0))):
            directory = tmp_path / "data" / split / class_name
            directory.mkdir(parents=True)
            Image.new("RGB", (16, 16), color).save(directory / "image.png")
    Image.new("RGB", (16, 16), (255, 0, 0)).save(tmp_path / "data" / "train" / "zero" / "tail.png")
    task = replace(
        _task(tmp_path, classes=2),
        epochs=1,
        workers=workers,
        execution=runtime.ExecutionOptions(prefetch_factor=2),
    )
    bundles: list[runtime.LoaderBundle] = []

    def build_loaders(
        active: runtime.BaselineTask, device: torch.device
    ) -> runtime.LoaderBundle:
        bundle = runtime._build_loaders(active, device)
        bundles.append(bundle)
        return bundle

    result = runtime.run_task(
        task,
        device=torch.device("cpu"),
        model_builder=_builder,
        loader_builder=build_loaders,
    )

    assert task.checkpoint_path.is_file()
    assert task.result_path.is_file()
    assert result["completed_epochs"] == 1
    assert result["history"][0]["train"]["examples"] == 3.0
    assert result["final_validation"]["examples"] == 2.0
    if workers:
        assert bundles[0].train.prefetch_factor == 2
        assert bundles[0].validation.prefetch_factor == 2
        assert not bundles[0].train.persistent_workers
        assert not bundles[0].validation.persistent_workers


def test_historical_checkpoint_contract_is_rejected_without_unsafe_resume(tmp_path: Path) -> None:
    task = _task(tmp_path, resume=True)
    task.checkpoint_path.parent.mkdir(parents=True)
    torch.save({"schema": runtime.SCHEMA, "contract_sha256": "old"}, task.checkpoint_path)
    contract = runtime._contract(task, torch.device("cpu"))
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    runtime._atomic_json(contract_path, contract)

    with pytest.raises(RuntimeError, match="incompatible historical"):
        runtime.run_task(
            task,
            device=torch.device("cpu"),
            model_builder=_builder,
            loader_builder=_loader,
            mixup_factory=lambda active: IdentityMixup(active.num_classes),
        )


def test_invalid_artifacts_are_rejected_before_writing_a_contract(tmp_path: Path) -> None:
    task = _task(tmp_path, resume=True)
    task.checkpoint_path.parent.mkdir(parents=True)
    torch.save({"schema": runtime.SCHEMA, "contract_sha256": "old"}, task.checkpoint_path)

    with pytest.raises(RuntimeError, match="checkpoint has no contract"):
        runtime.run_task(
            task,
            device=torch.device("cpu"),
            model_builder=_builder,
            loader_builder=_loader,
            mixup_factory=lambda active: IdentityMixup(active.num_classes),
        )
    assert not (task.output_dir / "contracts").exists()


def test_partial_checkpoint_and_result_are_not_treated_as_complete(tmp_path: Path) -> None:
    task = _task(tmp_path, resume=True)
    contract = runtime._contract(task, torch.device("cpu"))
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    runtime._atomic_json(contract_path, contract)
    task.checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "schema": runtime.SCHEMA,
            "contract_sha256": runtime._sha256_payload(contract),
            "task_sha256": contract["task_sha256"],
            "source_digest_sha256": contract["source_digest_sha256"],
            "completed_epochs": 1,
            "global_step": 0,
            "history": [],
            "model": {},
            "optimizer": {},
            "rng": {},
        },
        task.checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="structurally incomplete"):
        runtime.run_task(
            task,
            device=torch.device("cpu"),
            model_builder=_builder,
            loader_builder=_loader,
            mixup_factory=lambda active: IdentityMixup(active.num_classes),
        )

    result_task = replace(task, resume=False)
    result_task.result_path.parent.mkdir(parents=True)
    runtime._atomic_json(
        result_task.result_path,
        {
            "schema": runtime.SCHEMA,
            "contract_sha256": runtime._sha256_payload(contract),
            "task_sha256": contract["task_sha256"],
            "source_digest_sha256": contract["source_digest_sha256"],
            "status": "completed",
        },
    )
    with pytest.raises(RuntimeError, match="structurally incomplete"):
        runtime.run_task(
            result_task,
            device=torch.device("cpu"),
            model_builder=_builder,
            loader_builder=_loader,
            mixup_factory=lambda active: IdentityMixup(active.num_classes),
        )


def test_cpu_rng_capture_does_not_query_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _loader(_task(tmp_path), torch.device("cpu"))
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("CUDA was touched"),
    )

    assert runtime._capture_rng(bundle, torch.device("cpu"))["cuda"] == []


def test_online_mode_uses_an_injected_logger_without_network(tmp_path: Path) -> None:
    records: list[dict[str, object]] = []

    class Logger:
        def log(self, record: dict[str, object]) -> None:
            records.append(record)

        def finish(self, result: dict[str, object]) -> None:
            records.append({"kind": "finished", **result})

    task = replace(_task(tmp_path), epochs=1, wandb_mode="online")
    runtime.run_task(
        task,
        device=torch.device("cpu"),
        model_builder=_builder,
        loader_builder=_loader,
        mixup_factory=lambda active: IdentityMixup(active.num_classes),
        logger_factory=lambda _task, _contract: Logger(),
    )

    assert [record["kind"] for record in records] == ["epoch", "finished"]
    contract = runtime._contract(task, torch.device("cpu"))
    assert contract["runtime"]["device_type"] == "cpu"


def test_online_sdk_run_id_is_stable_for_resume_and_distinct_for_new_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def initialize(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    task = replace(_task(tmp_path / "first"), wandb_mode="online")
    for active in (
        task,
        replace(task, resume=True),
        replace(_task(tmp_path / "second"), wandb_mode="online"),
    ):
        runtime._make_logger(active, runtime._contract(active, torch.device("cpu")))

    assert calls[0]["id"] == calls[1]["id"]
    assert calls[0]["id"] != calls[2]["id"]
    assert all(call["resume"] == "allow" and call["mode"] == "online" for call in calls)


def test_online_sdk_is_closed_even_when_final_log_fails() -> None:
    exit_codes: list[int] = []

    class BrokenRun:
        def log(self, _record: dict[str, object]) -> None:
            raise RuntimeError("telemetry unavailable")

        def finish(self, exit_code: int = 0) -> None:
            exit_codes.append(exit_code)

    with pytest.raises(RuntimeError, match="telemetry unavailable"):
        runtime.WandbLogger(BrokenRun()).finish({"status": "failed"})
    assert exit_codes == [1]


def test_logger_finishes_when_training_is_interrupted(tmp_path: Path) -> None:
    records: list[dict[str, object]] = []

    class Logger:
        def log(self, record: dict[str, object]) -> None:
            records.append(record)

        def finish(self, result: dict[str, object]) -> None:
            records.append(result)

    def interrupted_loader(
        _task: runtime.BaselineTask,
        _device: torch.device,
    ) -> runtime.LoaderBundle:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runtime.run_task(
            _task(tmp_path),
            device=torch.device("cpu"),
            model_builder=_builder,
            loader_builder=interrupted_loader,
            mixup_factory=lambda active: IdentityMixup(active.num_classes),
            logger_factory=lambda _task, _contract: Logger(),
        )

    assert records == [{"kind": "interrupted", "status": "interrupted", "model_key": "tiny"}]


def test_runtime_source_digest_cannot_be_overridden_by_task_metadata(tmp_path: Path) -> None:
    task = replace(_task(tmp_path), source_sha256={"imagenet1k_runtime": "forged"})

    source_sha256 = runtime._contract(task, torch.device("cpu"))["source_sha256"]
    assert source_sha256["imagenet1k_runtime"] != "forged"


def test_execution_options_are_contract_bound_and_cpu_fallback_trains(
    tmp_path: Path,
) -> None:
    options = runtime.ExecutionOptions(
        compile_model=False,
        gpu_mixup=True,
        non_blocking_transfer=True,
        prefetch_factor=2,
        metric_sync_interval=100,
    )
    task = replace(_task(tmp_path), epochs=1, execution=options)

    result = runtime.run_task(
        task,
        device=torch.device("cpu"),
        model_builder=_builder,
        loader_builder=_loader,
        mixup_factory=lambda active: IdentityMixup(active.num_classes),
    )

    contract = runtime._contract(task, torch.device("cpu"))
    assert contract["execution"] == {
        "compile_model": False,
        "gpu_mixup": True,
        "non_blocking_transfer": True,
        "prefetch_factor": 2,
        "metric_sync_interval": 100,
    }
    assert "execution" not in runtime._task_payload(task)
    assert result["history"][0]["train"]["examples"] == 8.0
    assert result["final_validation"]["examples"] == 4.0


def test_compiled_module_resume_preserves_state_names_and_optimizer_hooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runtime.torch._dynamo.config,
        "recompile_limit",
        runtime.torch._dynamo.config.recompile_limit,
    )
    compiled: list[tuple[nn.Module, tuple[object, ...], dict[str, object]]] = []
    updated: list[int] = []
    models: list[nn.Module] = []

    def compile_call(
        module_call: object, *args: object, **kwargs: object
    ) -> object:
        model = getattr(module_call, "__self__", None)
        assert isinstance(model, nn.Module)
        compiled.append((model, args, kwargs))
        # Keep the public Module.compile call path in place without running inductor.
        return module_call

    def build_model(_key: str, _source: str | Path | None, classes: int) -> nn.Module:
        model = TinyClassifier(classes)
        models.append(model)
        return model

    def optimizer_builder(
        model: nn.Module, learning_rate: float, device: torch.device
    ) -> torch.optim.Optimizer:
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        optimizer = runtime._build_optimizer(model, learning_rate, device)
        optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: updated.append(1)
        )
        assert parameter_ids == tuple(id(parameter) for parameter in model.parameters())
        return optimizer

    options = runtime.ExecutionOptions(compile_model=True, metric_sync_interval=100)
    task = replace(_task(tmp_path), execution=options)
    original_emit = runtime._emit_progress
    monkeypatch.setattr(runtime.torch, "compile", compile_call)
    interrupted = False

    def interrupt_after_first_epoch(
        _task: runtime.BaselineTask, payload: dict[str, object]
    ) -> None:
        nonlocal interrupted
        original_emit(_task, payload)
        if payload["epoch"] == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "_emit_progress", interrupt_after_first_epoch)
    # The first call compiles, trains and commits the first epoch before interruption.
    with pytest.raises(KeyboardInterrupt):
        runtime.run_task(
            task,
            device=torch.device("cpu"),
            model_builder=build_model,
            loader_builder=_loader,
            mixup_factory=lambda active: IdentityMixup(active.num_classes),
            optimizer_builder=optimizer_builder,
        )
    checkpoint = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    expected_names = set(models[-1].state_dict())
    assert set(checkpoint["model"]) == expected_names
    assert all("_orig_mod" not in name for name in checkpoint["model"])

    resumed = runtime.run_task(
        replace(task, resume=True),
        device=torch.device("cpu"),
        model_builder=build_model,
        loader_builder=_loader,
        mixup_factory=lambda active: IdentityMixup(active.num_classes),
        optimizer_builder=optimizer_builder,
    )

    assert resumed["completed_epochs"] == 2
    assert len(compiled) == 2
    assert all(
        not args and kwargs == {"mode": "default", "dynamic": False}
        for _, args, kwargs in compiled
    )
    assert all(
        compiled_module is model
        for (compiled_module, _, _), model in zip(compiled, models, strict=True)
    )
    assert len(updated) == 2


def test_compile_errors_are_not_silently_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime.torch._dynamo.config,
        "recompile_limit",
        runtime.torch._dynamo.config.recompile_limit,
    )

    def broken_compile(_model: nn.Module, **_kwargs: object) -> None:
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(runtime.nn.Module, "compile", broken_compile, raising=False)

    with pytest.raises(RuntimeError, match="compiler unavailable"):
        runtime._compile_model_in_place(
            TinyClassifier(3), runtime.ExecutionOptions(compile_model=True)
        )
    assert runtime.torch._dynamo.config.recompile_limit == 64


def test_module_compile_keeps_raw_parameter_objects_and_state_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime.torch._dynamo.config, "recompile_limit", 8)
    model = TinyClassifier(3)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state_names = tuple(model.state_dict())
    runtime._compile_model_in_place(model, runtime.ExecutionOptions())
    assert runtime.torch._dynamo.config.recompile_limit == 8

    runtime._compile_model_in_place(model, runtime.ExecutionOptions(compile_model=True))

    assert runtime.torch._dynamo.config.recompile_limit == 64
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert tuple(model.state_dict()) == state_names


def test_odd_tail_and_physical_batch_512_keep_all_examples_and_step_count(
    tmp_path: Path,
) -> None:
    task = replace(_task(tmp_path, classes=3), epochs=1)

    def odd_loader(
        active: runtime.BaselineTask, _device: torch.device
    ) -> runtime.LoaderBundle:
        generator = torch.Generator().manual_seed(19)
        train_dataset = TensorDataset(
            torch.randn(257, 3, 4, 4, generator=generator),
            torch.arange(257) % active.num_classes,
        )
        validation_dataset = TensorDataset(
            torch.randn(3, 3, 4, 4, generator=generator),
            torch.arange(3) % active.num_classes,
        )
        train_generator = torch.Generator().manual_seed(active.seed)
        validation_generator = torch.Generator().manual_seed(active.seed + 1)
        return runtime.LoaderBundle(
            DataLoader(
                train_dataset,
                batch_size=active.batch_size,
                shuffle=True,
                generator=train_generator,
            ),
            DataLoader(validation_dataset, batch_size=active.batch_size),
            train_generator,
            validation_generator,
        )

    result = runtime.run_task(
        task,
        device=torch.device("cpu"),
        model_builder=_builder,
        loader_builder=odd_loader,
        mixup_factory=lambda active: IdentityMixup(active.num_classes),
    )
    assert result["history"][0]["train"]["examples"] == 257.0
    assert result["history"][0]["train"]["optimizer_steps"] == 2.0

    wide_task = replace(task, batch_size=512, effective_batch_size=512, epochs=200)
    assert wide_task.gradient_accumulation_steps == 1
    assert (
        runtime._contract(wide_task, torch.device("cpu"))["recipe"]["effective_batch_size"]
        == 512
    )
    samples = TensorDataset(
        torch.randn(1024, 3, 4, 4),
        torch.arange(1024) % wide_task.num_classes,
    )
    wide_loader = DataLoader(samples, batch_size=wide_task.batch_size)
    model = TinyClassifier(wide_task.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=wide_task.learning_rate)
    train, global_step = runtime._train_epoch(
        model,
        wide_loader,
        optimizer,
        IdentityMixup(wide_task.num_classes),
        wide_task,
        torch.device("cpu"),
        global_step=50_050,
        epoch=11,
    )

    assert train["examples"] == 1024.0
    assert train["optimizer_steps"] == 2.0
    assert global_step == 50_052
    assert optimizer.param_groups[0]["lr"] == runtime._learning_rate(wide_task, 21, 2)
    assert optimizer.param_groups[0]["lr"] != runtime._learning_rate(wide_task, 50_051, 2)
