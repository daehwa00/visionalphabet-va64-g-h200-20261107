from __future__ import annotations

# pyright: reportArgumentType=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import queue_r2k3_lanes as queue
from scripts import run_a2d_r2k3_path_collapse_overnight_imagenet100 as runner


def test_overnight_queue_partitions_every_candidate_once() -> None:
    gpu0 = runner.JOBS_BY_GPU[0]
    gpu1 = runner.JOBS_BY_GPU[1]
    assert not set(gpu0).intersection(gpu1)
    assert set(gpu0).union(gpu1) == set(runner.VARIANTS)
    assert gpu0[:2] == (runner.PF2K_PATH_H4, runner.PF2K_PATH_PG_H4)
    assert gpu1[:2] == (runner.PF15K_PATH_H4, runner.PF15K_PATH_PG_H4)


def test_completed_result_requires_final_epoch_and_checkpoint(tmp_path: Path) -> None:
    variant = runner.PF15K_PATH_H4
    root = tmp_path / "run"
    result = root / "results" / f"{variant}__seed501.json"
    checkpoint = root / "checkpoints" / f"{variant}__seed501.pt"
    result.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    result.write_text(json.dumps({"history": [{"epoch": 99}]}) + "\n")
    checkpoint.touch()
    assert not queue._result_is_complete(root, variant, 100)
    result.write_text(json.dumps({"history": [{"epoch": 100}]}) + "\n")
    assert queue._result_is_complete(root, variant, 100)


def test_cpu_affinity_parser_keeps_physical_core_pairs() -> None:
    assert queue._parse_cpus("0-7,16-23") == set(range(8)).union(range(16, 24))


@pytest.mark.parametrize("specification", ["", "0,", "0,3-2", "-1", "one"])
def test_cpu_affinity_parser_rejects_invalid_ranges(specification: str) -> None:
    with pytest.raises(ValueError, match="invalid queue CPU affinity"):
        queue._parse_cpus(specification)


def test_compile_cache_is_shared_by_stack_and_gpu_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(queue, "_compile_stack_namespace", lambda: Path("stack-v1"))
    monkeypatch.setattr(queue, "_gpu_cache_identity", lambda _gpu: "sm120-pro6000")
    environment = {
        "LNET_COMPILE_CACHE_SCOPE": "shared",
        "LNET_COMPILE_CACHE_ROOT": str(tmp_path / "shared"),
    }
    first = queue._compile_cache_paths(
        SimpleNamespace(root=tmp_path / "run-a", gpu=0),
        environment,
    )
    second = queue._compile_cache_paths(
        SimpleNamespace(root=tmp_path / "run-b", gpu=0),
        environment,
    )
    expected = tmp_path / "shared/stack-v1/sm120-pro6000"
    assert first == second
    assert first["namespace"] == expected
    assert first["torchinductor"] == expected / "torchinductor"
    assert first["triton"] == expected / "triton"


def test_compile_cache_can_be_isolated_for_debugging(tmp_path: Path) -> None:
    args = SimpleNamespace(root=tmp_path / "run", gpu=0)
    cache = queue._compile_cache_paths(
        args,
        {"LNET_COMPILE_CACHE_SCOPE": "isolated"},
    )
    assert cache["namespace"] == args.root / "cache"


def test_valid_smoke_accepts_runner_representatives(tmp_path: Path) -> None:
    fingerprint = "f" * 64
    evidence = tmp_path / "evidence/cuda/smoke.json"
    marker = tmp_path / "evidence/cuda/source.sha256"
    evidence.parent.mkdir(parents=True)
    representatives = ("small", "large")
    evidence.write_text(
        json.dumps(
            {
                "status": "passed",
                "device": "cuda",
                "candidate_count": 2,
                "variants": {
                    name: {"status": "passed", "checkpoint_max_abs": 0.0}
                    for name in representatives
                },
                "full_batch": {
                    name: {"steps": 2.0, "peak_bytes": 1024.0} for name in representatives
                },
            }
        )
    )
    marker.write_text(fingerprint + "\n")
    args = SimpleNamespace(
        root=tmp_path,
        runner=SimpleNamespace(
            VARIANTS=("small", "medium", "large"),
            SMOKE_VARIANTS=representatives,
        ),
    )

    assert queue._valid_smoke(args, fingerprint)


def test_queue_environment_exports_shared_compile_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "shared/stack/gpu"
    cache = {
        "namespace": namespace,
        "torchinductor": namespace / "torchinductor",
        "triton": namespace / "triton",
    }
    monkeypatch.setattr(queue, "_compile_cache_paths", lambda *_args: cache)
    args = SimpleNamespace(
        gpu=0,
        repo=tmp_path / "runtime",
        root=tmp_path / "run",
        workers=8,
        wandb_project="project",
        wandb_entity="entity",
        wandb_group="group",
    )
    environment = queue._environment(args, "candidate")
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == str(cache["torchinductor"])
    assert environment["TRITON_CACHE_DIR"] == str(cache["triton"])
    assert environment["LNET_COMPILE_CACHE_NAMESPACE"] == str(namespace)
    assert cache["torchinductor"].is_dir()
    assert cache["triton"].is_dir()


def test_gpu_cache_identity_uses_capability_and_product_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(stdout="12.0, NVIDIA RTX PRO 6000 Blackwell Server Edition\n")
    monkeypatch.setattr(queue.subprocess, "run", lambda *_args, **_kwargs: completed)
    assert queue._gpu_cache_identity(0) == ("sm120-NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition")


def test_restarted_final_attempt_rechecks_surviving_child_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    variant = runner.PF15K_PATH_H4
    run_root = tmp_path / "runs" / variant
    state = {
        "variant": variant,
        "status": "QUEUED",
        "attempts": 2,
        "run_root": str(run_root),
        "wandb_url": None,
        "started_at": "already-started",
        "ended_at": None,
        "last_error": None,
    }
    status = {"jobs": {variant: state}}
    args = SimpleNamespace(root=tmp_path, gpu=0, epochs=100, max_attempts=2)
    checks = iter((False, True))
    waits: list[Path] = []
    writes: list[str] = []

    monkeypatch.setattr(queue, "_result_is_complete", lambda *_args: next(checks))
    monkeypatch.setattr(queue, "_wait_for_gpu", lambda _args, path: waits.append(path))
    monkeypatch.setattr(
        queue,
        "_write_status",
        lambda _args, _status: writes.append(state["status"]),
    )
    monkeypatch.setattr(
        queue,
        "_run_logged",
        lambda *_args, **_kwargs: pytest.fail("a completed surviving child was relaunched"),
    )

    queue._run_job(args, variant, status)

    assert len(waits) == 1
    assert state["attempts"] == 2
    assert state["status"] == "COMPLETE"
    assert state["ended_at"] is not None
    assert writes == ["COMPLETE"]


def test_exhausted_attempt_waits_once_then_fails_without_relaunch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    variant = runner.PF15K_PATH_H4
    state = {
        "variant": variant,
        "status": "QUEUED",
        "attempts": 2,
        "run_root": str(tmp_path / "runs" / variant),
        "wandb_url": None,
        "started_at": "already-started",
        "ended_at": None,
        "last_error": "prior failure",
    }
    status = {"jobs": {variant: state}}
    args = SimpleNamespace(root=tmp_path, gpu=0, epochs=100, max_attempts=2)
    waits: list[Path] = []

    monkeypatch.setattr(queue, "_result_is_complete", lambda *_args: False)
    monkeypatch.setattr(queue, "_wait_for_gpu", lambda _args, path: waits.append(path))
    monkeypatch.setattr(queue, "_write_status", lambda *_args: None)
    monkeypatch.setattr(
        queue,
        "_run_logged",
        lambda *_args, **_kwargs: pytest.fail("an exhausted attempt was relaunched"),
    )

    queue._run_job(args, variant, status)

    assert len(waits) == 1
    assert state["attempts"] == 2
    assert state["status"] == "FAILED"
    assert state["ended_at"] is not None
