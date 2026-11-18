"""Run VA64 with verified seed-0 initialization and GPU Mixup."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn


def load_initial(model: nn.Module, path: Path, sha: str, variant: str, source_sha: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError("initial state byte hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("trained") is not False
        or payload.get("seed") != 0
        or payload.get("variant") != variant
    ):
        raise RuntimeError("not a matching untrained seed-0 initialization")
    if payload.get("source_manifest_sha256") != source_sha:
        raise RuntimeError("initial state source mismatch")
    model.load_state_dict(payload["model"], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), payload["model"][name]):
            raise RuntimeError("initial tensor load was not exact")


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--initial-state", type=Path, required=True)
    p.add_argument("--initial-state-sha256", required=True)
    extra, remaining = p.parse_known_args()
    sys.path[:0] = [str(extra.code / "src"), str(extra.code / "scripts")]
    import run_va64_retrain_imagenet1k as experiment

    sys.argv = [sys.argv[0], *remaining]
    args = experiment.arguments()
    source_sha = hashlib.sha256((extra.code / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    original_build = experiment.build_model
    original_task = experiment.make_task

    def paired_build(variant: str, *, prepared: bool = True) -> nn.Module:
        model = original_build(variant, prepared=prepared)
        load_initial(model, extra.initial_state, extra.initial_state_sha256, variant, source_sha)
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
                "canonical_seed0_initial_state": extra.initial_state_sha256,
                "fast_gpu_mixup_entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )

    experiment.build_model = paired_build
    experiment.make_task = paired_task
    experiment.train(args)


if __name__ == "__main__":
    main()
