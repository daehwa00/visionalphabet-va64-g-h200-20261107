"""G training paired with each seed's untrained canonical B backbone."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import run_va64_retrain_imagenet1k as experiment
import torch

INITIAL_SHA = "4fe26280065838b6b0da66b6a2ff5082ac9be55a9a661a9296bdf10cb514de94"
INITIAL_SHAS = {
    0: INITIAL_SHA,
    1: "4eec30c5a59ea1f2a2d1ebfdb33d35f37a04c4d656c52b1174bcbf0300eea01e",
    2: "987f7a8b71eb6481c8f3d42aa54b7da969247edd7fd5d3b0b8a2db4fbe70311b",
}


def initialization_sha(seed: int) -> str:
    if seed not in INITIAL_SHAS:
        raise ValueError("G canonical initialization supports seeds 0, 1, and 2 only")
    # Keep the original seed-0 constant usable by existing callers/tests.
    return INITIAL_SHA if seed == 0 else INITIAL_SHAS[seed]


def initialize_common_backbone(model, path=None, seed=0):
    expected_sha = initialization_sha(seed)
    path = (
        Path(path)
        if path is not None
        else (Path(__file__).resolve().parents[1] / "initialization" / f"initial-B{seed}.pt")
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError("canonical initialization hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("variant") != "B"
        or payload.get("seed") != seed
        or payload.get("trained") is not False
    ):
        raise RuntimeError(f"not the untrained B{seed} initialization")
    current = model.state_dict()
    shared = {name: value for name, value in payload["model"].items() if ".classifier." not in name}
    expected = {name for name in current if ".classifier." not in name}
    if set(shared) != expected:
        raise RuntimeError("canonical backbone structure mismatch")
    current.update(shared)
    model.load_state_dict(current, strict=True)
    return model


def validate_run(args):
    if args.variant != "G":
        raise ValueError("this paired entrypoint is G only")
    initialization_sha(args.seed)
    if args.seed != 0 and (
        args.resume or os.environ.get("VA64_G_RECOVERY") or os.environ.get("VA64_G_RESUME_PLAN")
    ):
        raise ValueError("G seeds 1 and 2 require a fresh start; recovery is seed-0 only")


def main():
    args = experiment.arguments()
    validate_run(args)
    original_build = experiment.build_model
    original_task = experiment.make_task

    def build(variant, *, prepared=True):
        return initialize_common_backbone(
            original_build(variant, prepared=prepared), seed=args.seed
        )

    def task(arguments):
        value = original_task(arguments)
        return dataclasses.replace(
            value,
            execution=dataclasses.replace(value.execution, gpu_mixup=True),
            source_sha256={
                **value.source_sha256,
                "canonical_backbone_initialization": initialization_sha(arguments.seed),
                "h200_g_entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )

    experiment.build_model = build
    experiment.make_task = task
    if os.environ.get("VA64_G_MANAGED_ROOT"):
        os.environ["VA64_G_JOB"] = f"va64-g-seed{args.seed}"
        from va64_runtime_operations import install_training

        install_training(experiment.runtime, Path(os.environ["VA64_G_MANAGED_ROOT"]))
    if os.environ.get("VA64_G_RECOVERY"):
        from va64_h200_resume import migrate

        original_run = experiment.runtime.run_task

        def migrated_run(task, **kwargs):
            migrate(
                task,
                Path(os.environ["VA64_G_RECOVERY"]),
                Path(os.environ["VA64_G_RESUME_PLAN"]),
                torch.device("cuda"),
            )
            return original_run(dataclasses.replace(task, resume=True), **kwargs)

        experiment.runtime.run_task = migrated_run
    try:
        result = experiment.train(args)
        if os.environ.get("VA64_G_MANAGED_ROOT"):
            from va64_runtime_operations import complete_run

            complete_run(
                Path(os.environ["VA64_G_MANAGED_ROOT"]),
                experiment.make_task(args).output_dir,
                result["status"],
            )
    except experiment.runtime.TrainingInterrupted:
        if os.environ.get("VA64_G_MANAGED_ROOT"):
            from va64_runtime_operations import current

            root = Path(os.environ["VA64_G_MANAGED_ROOT"])
            current(
                root,
                os.environ.get("VA64_G_JOB", f"va64-g-seed{args.seed}"),
                "stopped_without_backup",
                root / "runs" / "va_k64_retrain_G" / f"seed_{args.seed}",
            )
        print("VA64_STOPPED_WITHOUT_VALIDATION_OR_BACKUP", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
