from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
import json
import random
from argparse import Namespace
from pathlib import Path

import a2d_r2k3_run_artifacts as artifacts
import a2d_r2k3_runtime as runtime
import numpy as np
import pytest
import queue_a2d_r2k3_same_resolution_factorial as queue
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner
import smoke_a2d_r2k3_same_resolution_factorial as smoke
import torch


def _queue_args(tmp_path: Path) -> Namespace:
    return Namespace(
        gpu=0,
        repo=Path(__file__).resolve().parents[1],
        root=tmp_path,
        data_root=tmp_path / "data",
        python=tmp_path / "python",
        epochs=100,
        batch_size=128,
        workers=8,
        max_attempts=2,
        wandb_project="project",
        wandb_entity="entity",
        wandb_group="group",
    )


def _write_valid_run(
    run_root: Path,
    variant: str,
    epochs: int = 100,
    *,
    source_sha256: dict[str, str] | None = None,
    exact_model: bool = False,
) -> None:
    result = run_root / "results" / f"{variant}__seed501.json"
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    result.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    global_step = epochs * 10
    recipe = {
        "epochs": epochs,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "precision": "bfloat16",
        "loader_workers": 8,
        "loader_persistent_workers": True,
        "learning_rate": 3.0e-3,
        "weight_decay": 0.05,
        "modal_learning_rate_multiplier": 1.0,
        "pole_geometry_learning_rate_multiplier": 1.0,
        "fused_optimizer": False,
    }
    if exact_model:
        model = runner._build(variant, runtime.model_config())
        optimizer = runtime.build_optimizer(model, recipe)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        scheduler_state["last_epoch"] = epochs
        scheduler_state["_step_count"] = epochs + 1
    else:
        model_state = {"weight": torch.ones(1)}
        optimizer_state = {
            "state": {0: {"step": torch.tensor(1.0)}},
            "param_groups": [{"params": [0]}],
        }
        scheduler_state = {"base_lrs": [0.1], "last_epoch": epochs, "_step_count": epochs + 1}
    torch.save(
        {
            "variant": variant,
            "seed": 501,
            "epoch": epochs,
            "global_step": global_step,
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "history": [{"epoch": epoch} for epoch in range(1, epochs + 1)],
            "training_seconds": 1.0,
            "training_generator_state": torch.Generator().get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": [
                torch.Generator(device="cuda").get_state()
                if torch.cuda.is_available()
                else torch.zeros(16, dtype=torch.uint8)
            ],
            "python_rng_state": random.Random(501).getstate(),  # noqa: S311
            "mixup_rng_state": np.random.default_rng(501).bit_generator.state,
        },
        checkpoint,
    )
    result.write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": 501,
                "parameters": 123,
                "global_step": global_step,
                "history": [{"epoch": epoch} for epoch in range(1, epochs + 1)],
                "final_validation": {"accuracy": 0.5, "cross_entropy": 1.0},
            }
        )
    )
    (run_root / "contract.json").write_text(
        json.dumps(
            {
                "variants": [variant],
                "seeds": [501],
                "recipe": recipe,
                "data": {
                    "manifest_sha256": "data-manifest",
                    "train_images": 100,
                    "validation_images": 20,
                },
                "source_sha256": source_sha256 or {},
            }
        )
    )


def test_queue_preflight_sources_are_available() -> None:
    repo = Path(__file__).resolve().parents[1]
    sources = tuple(queue._source_paths(repo))
    assert all(path.is_file() for path in sources)
    assert repo / "scripts/smoke_a2d_r2k3_same_resolution_factorial.py" in sources
    assert repo / "scripts/a2d_r2k3_runtime.py" in sources
    assert repo / "scripts/a2d_r2k3_capacity_factory.py" in sources
    assert repo / "scripts/run_alphabet2d_imagenet100_nano.py" in sources
    assert repo / "scripts/benchmark_generic_complex_ffn.py" not in sources
    assert callable(smoke.main)


def test_smoke_subset_requires_its_width_matched_base() -> None:
    base = runner.variant_name(96, ())
    maximal = runner.variant_name(96, runner.RESOLUTIONS)

    smoke._validate_variant_selection((base, maximal))
    with pytest.raises(ValueError, match=base):
        smoke._validate_variant_selection((maximal,))


def test_smoke_preflight_is_serialized_by_campaign_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        queue,
        "_ensure_smoke",
        lambda _args, fingerprint, log: calls.append((fingerprint, log)),
    )
    log = tmp_path / "preflight.log"

    queue._ensure_smoke_once(Namespace(root=tmp_path), "source-sha", log)

    assert calls == [("source-sha", log)]
    assert (tmp_path / "smoke.lock").is_file()


def test_queue_resume_rejects_recipe_changes(tmp_path: Path) -> None:
    args = _queue_args(tmp_path)
    status = queue._initial_status(args, "source-sha")
    (tmp_path / "queue-gpu0.json").write_text(json.dumps(status))
    changed = _queue_args(tmp_path)
    changed.batch_size = 64

    with pytest.raises(RuntimeError, match="mixed-recipe"):
        queue._load_status(changed, "source-sha")


def test_queue_resume_revalidates_completed_artifacts(tmp_path: Path) -> None:
    args = _queue_args(tmp_path)
    status = queue._initial_status(args, "source-sha")
    variant = next(iter(status["jobs"]))
    status["jobs"][variant]["status"] = "COMPLETE"
    (tmp_path / "queue-gpu0.json").write_text(json.dumps(status))

    loaded = queue._load_status(args, "source-sha")

    assert loaded["jobs"][variant]["status"] == "QUEUED"
    assert "artifact validation failed" in loaded["jobs"][variant]["last_error"]


def test_run_and_reference_validation_require_complete_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    variant = "K128-PF15K-D1"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant)
    references = {
        name: {
            "existing_variant": variant,
            "run_root": str(run_root),
        }
        for name in runner.REFERENCE_VARIANTS
    }
    monkeypatch.setattr(queue, "_reference_state", lambda: references)
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_payload.pop("global_step")
    torch.save(checkpoint_payload, checkpoint)

    assert artifacts.valid_run(run_root, variant, 100)
    queue._validate_references(references)

    checkpoint.unlink()
    assert not artifacts.valid_run(run_root, variant, 100)
    with pytest.raises(RuntimeError, match="references are incomplete"):
        queue._validate_references(references)

    with pytest.raises(RuntimeError, match="reference mapping changed"):
        queue._validate_references({})


def test_run_validation_binds_campaign_data_and_source(tmp_path: Path) -> None:
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    source_sha256 = {"r2k3_dependency_tree": "source-tree"}
    _write_valid_run(run_root, variant, source_sha256=source_sha256, exact_model=True)
    campaign = {
        "epochs": 100,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "precision": "bfloat16",
        "loader_workers": 8,
        "loader_persistent_workers": True,
        "data_manifest_sha256": "data-manifest",
        "train_images": 100,
        "validation_images": 20,
    }

    assert artifacts.valid_run(
        run_root,
        variant,
        100,
        campaign=campaign,
        source_sha256=source_sha256,
    )


def test_invalid_completion_is_quarantined_for_a_clean_retry(tmp_path: Path) -> None:
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant)
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    result = run_root / "results" / f"{variant}__seed501.json"
    checkpoint.write_bytes(b"not a checkpoint")
    campaign = {
        "epochs": 100,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "precision": "bfloat16",
    }

    assert not artifacts.valid_run(run_root, variant, 100)
    artifacts.quarantine_invalid_completion(
        run_root,
        variant,
        100,
        campaign=campaign,
        source_sha256={},
    )

    assert not result.exists()
    assert not checkpoint.exists()
    assert list(result.parent.glob(f"{result.stem}.invalid-*.json"))
    assert list(checkpoint.parent.glob(f"{checkpoint.stem}.invalid-*.pt"))


def test_resumable_checkpoint_requires_full_restore_state(tmp_path: Path) -> None:
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant)
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        require_global_step=True,
    )
    payload.pop("global_step")
    assert artifacts.checkpoint_is_resumable(payload, variant, 100, require_final=False)
    assert not artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        require_global_step=True,
    )
    payload["global_step"] = 1000
    payload["training_generator_state"] = torch.zeros(1, dtype=torch.uint8)
    assert not artifacts.checkpoint_is_resumable(payload, variant, 100, require_final=True)
    payload["training_generator_state"] = torch.Generator().get_state()
    payload["optimizer"]["state"] = {}
    assert not artifacts.checkpoint_is_resumable(payload, variant, 100, require_final=True)
    payload["optimizer"]["state"] = {0: 1}
    assert not artifacts.checkpoint_is_resumable(payload, variant, 100, require_final=True)


def test_exact_optimizer_validation_requires_all_adam_slots(tmp_path: Path) -> None:
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant, exact_model=True)
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recipe = json.loads((run_root / "contract.json").read_text())["recipe"]

    assert artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        validate_model=True,
        require_global_step=True,
        recipe=recipe,
    )
    first_state = next(iter(payload["optimizer"]["state"].values()))
    first_state.pop("exp_avg")
    assert not artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        validate_model=True,
        require_global_step=True,
        recipe=recipe,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["optimizer"]["state"].pop(next(iter(payload["optimizer"]["state"])))
    assert not artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        validate_model=True,
        require_global_step=True,
        recipe=recipe,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["scheduler"]["base_lrs"] = payload["scheduler"]["base_lrs"][:1]
    assert not artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        validate_model=True,
        require_global_step=True,
        recipe=recipe,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["optimizer"]["param_groups"][0].pop("betas")
    assert not artifacts.checkpoint_is_resumable(
        payload,
        variant,
        100,
        require_final=True,
        validate_model=True,
        require_global_step=True,
        recipe=recipe,
    )


def test_cuda_rng_validation_rejects_invalid_same_size_state(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA generator validation requires CUDA")
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant)
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["cuda_rng_state"] = [torch.ones_like(torch.Generator(device="cuda").get_state())]

    assert not artifacts.checkpoint_is_resumable(payload, variant, 100, require_final=True)


def test_orphan_corrupt_checkpoint_is_quarantined_without_a_result(tmp_path: Path) -> None:
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant)
    result = run_root / "results" / f"{variant}__seed501.json"
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    result.unlink()
    checkpoint.write_bytes(b"not a checkpoint")
    campaign = {
        "epochs": 100,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "precision": "bfloat16",
    }

    artifacts.quarantine_invalid_completion(
        run_root,
        variant,
        100,
        campaign=campaign,
        source_sha256={},
    )

    assert not checkpoint.exists()
    assert list(checkpoint.parent.glob(f"{checkpoint.stem}.invalid-*.pt"))


def test_stale_contract_quarantines_final_checkpoint(tmp_path: Path) -> None:
    variant = "K96-PF15K-Base"
    run_root = tmp_path / variant
    _write_valid_run(run_root, variant)
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    contract = run_root / "contract.json"
    campaign = {
        "epochs": 100,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "precision": "bfloat16",
    }

    artifacts.quarantine_invalid_completion(
        run_root,
        variant,
        100,
        campaign=campaign,
        source_sha256={"r2k3_runtime": "new-source"},
    )

    assert not contract.exists()
    assert not checkpoint.exists()


def test_predecessor_gate_waits_for_a_terminal_queue_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(json.dumps({"status": "RUNNING"}))
    log = tmp_path / "logs" / "predecessor.jsonl"

    def finish_predecessor(_seconds: int) -> None:
        predecessor.write_text(json.dumps({"status": "COMPLETE"}))

    monkeypatch.setattr(queue.time, "sleep", finish_predecessor)
    queue._wait_for_predecessor(
        Namespace(predecessor_status=predecessor, poll_seconds=0),
        log,
    )

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "predecessor_wait",
        "predecessor_complete",
    ]
    assert events[-1]["status"] == "COMPLETE"
