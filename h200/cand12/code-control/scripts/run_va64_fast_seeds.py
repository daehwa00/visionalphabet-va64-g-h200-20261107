"""Use canonical seed-1/2 initialization and the retained fast VA64 recipe."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn


def load_initial(model: nn.Module, path: Path, sha: str, variant: str, source_sha: str, seed: int) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError("initial state byte hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("trained") is not False
        or payload.get("seed") != seed
        or payload.get("variant") != variant
    ):
        raise RuntimeError("not a matching untrained seed initialization")
    if payload.get("source_manifest_sha256") != source_sha:
        raise RuntimeError("initial state source mismatch")
    model.load_state_dict(payload["model"], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), payload["model"][name]):
            raise RuntimeError("initial tensor load was not exact")


def parse_followup_arguments(argv: list[str], variants) -> argparse.Namespace:
    # The retained scientific runner deliberately accepts only seed 0. Keep its
    # immutable source intact; this follow-up adapter owns the seed-1/2 CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=variants, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    for name in ("data-root", "data-manifest", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--initial-state", type=Path, required=True)
    p.add_argument("--initial-state-sha256", required=True)
    p.add_argument("--validate-only", action="store_true")
    extra, remaining = p.parse_known_args()
    sys.path[:0] = [str(extra.code / "src"), str(extra.code / "scripts")]
    import run_va64_retrain_imagenet1k as experiment

    args = parse_followup_arguments(remaining, experiment.DEFINITIONS)
    if args.seed not in (1, 2):
        raise ValueError("follow-up entrypoint accepts only seeds 1 and 2")
    source_sha = hashlib.sha256((extra.code / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    original_build = experiment.build_model
    original_task = experiment.make_task

    def paired_build(variant: str, *, prepared: bool = True) -> nn.Module:
        model = original_build(variant, prepared=prepared)
        load_initial(model, extra.initial_state, extra.initial_state_sha256, variant, source_sha, args.seed)
        if variant == "P":
            experiment.assert_zero_phases(model)
        return model

    def paired_task(arguments: argparse.Namespace) -> experiment.runtime.BaselineTask:
        task = original_task(arguments)
        return dataclasses.replace(
            task,
            execution=dataclasses.replace(task.execution, gpu_mixup=True),
            source_sha256={
                **task.source_sha256,
                "canonical_initial_state": extra.initial_state_sha256,
                "fast_gpu_mixup_entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )

    experiment.build_model = paired_build
    experiment.make_task = paired_task
    if extra.validate_only:
        import json
        task = experiment.make_task(args)
        model = experiment.build_model(args.variant)
        if task.seed != args.seed or task.effective_batch_size != 256 or not task.execution.gpu_mixup:
            raise RuntimeError("follow-up recipe changed")
        print("VA64_FOLLOWUP_VALIDATED=" + json.dumps({
            "variant": args.variant, "seed": args.seed, "batch_size": task.batch_size,
            "accumulation": task.gradient_accumulation_steps, "epochs": task.epochs,
            "gpu_mixup": task.execution.gpu_mixup,
            "parameters": sum(value.numel() for value in model.parameters()),
            "initial_state_sha256": extra.initial_state_sha256,
        }), flush=True)
        return
    experiment.train(args)


if __name__ == "__main__":
    main()
