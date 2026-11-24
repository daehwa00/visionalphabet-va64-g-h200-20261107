"""Select and reuse a microbatch for one logical optimizer update."""

from __future__ import annotations

import fcntl
import socket
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch
from automatic_training_layout import (
    AutomaticTrainingRuntime,
    cpu_tree,
    digest,
    read_json,
    state_hash,
)
from effective_batch_search import baseline_microbatches, measure, valid_receipt
from effective_batch_update import run_update
from imagenet100_checkpoint_runtime import atomic_json

from lnet.model_w36b import compile_model_w36b_update

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | torch.Tensor]


class EffectiveTrainingRuntime(AutomaticTrainingRuntime):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decisions = {}
        self.fallbacks = {}
        self.compiled_update = compile_model_w36b_update(self.model, self.optimizer)

    def saved(self, key: str) -> dict[str, Any] | None:
        doc = read_json(self.base / (key + ".json"))
        if valid_receipt(doc, key):
            self.event(key, "reused", microbatch=doc["microbatch"], decision=doc["status"])
            return doc
        return None

    def configure(self, batch: Batch) -> tuple[str, dict[str, Any]]:
        x, y, other, mix = batch
        device = x.device
        props = torch.cuda.get_device_properties(device)
        fraction = torch.cuda.get_per_process_memory_fraction(device)
        free, total = torch.cuda.mem_get_info(device)
        limit = total * fraction
        available = min(free + torch.cuda.memory_reserved(device), limit)
        # Round driver bookkeeping away when the GPU is otherwise unoccupied.
        budget = limit if available >= 0.95 * limit else available
        info = {
            **self.identity,
            "effective_samples": len(y),
            "batch_layout": [
                (list(value.shape), str(value.dtype), list(value.stride()))
                for value in (x, y, other, mix)
                if torch.is_tensor(value)
            ],
            "loader_batch": self.recipe["batch_size"],
            "schedule_phase": (
                "warmup" if self.epoch_index < self.recipe.get("warmup_epochs", 0) else "scheduled"
            ),
            "zero_lr_groups": [group["lr"] == 0 for group in self.optimizer.param_groups],
            "memory_fraction": fraction,
            "memory_budget_gib": int(budget // 2**30),
        }
        key = digest(info)
        if key in self.decisions:
            return key, self.decisions[key]
        decision = self.saved(key)
        if decision is None:
            fallback = self.fallbacks.get(key)
            if fallback is not None and fallback[0] == self.epoch_index:
                return key, fallback[1]
            self.base.mkdir(parents=True, exist_ok=True)
            gpu = str(getattr(props, "uuid", None) or props.name)
            lock_name = "gpu-" + digest((socket.gethostname(), gpu)) + ".lock"
            with (self.base / (key + ".lock")).open("a") as workload_lock:
                fcntl.flock(workload_lock, fcntl.LOCK_EX)
                decision = self.saved(key)
                if decision is None:
                    with (self.base / lock_name).open("a") as gpu_lock:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                        decision = self._calibrate(batch, key, info, device)
        if decision["status"] == "retryable-fallback":
            self.fallbacks[key] = self.epoch_index, decision
        else:
            self.decisions[key] = decision
            self.fallbacks.pop(key, None)
        return key, decision

    def _calibrate(
        self,
        batch: Batch,
        key: str,
        info: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        self.event(key, "calibrating", effective_samples=len(batch[1]))
        live = self.live_state(device)
        before = {name: state_hash(value) for name, value in live.items()}
        snapshot = {
            "variant": self.variant,
            "model_config": self.model_config,
            "factories": self.factories,
            "recipe": self.recipe,
            "settings": self.settings,
            "model": cpu_tree(live["model"]),
            "optimizer": cpu_tree(live["optimizer"]),
            "rng": cpu_tree(live["rng"]),
            "batch": cpu_tree(batch),
            "index": device.index if device.index is not None else torch.cuda.current_device(),
            "memory_fraction": info["memory_fraction"],
        }
        problem = None
        with tempfile.TemporaryDirectory(prefix=".calibration-", dir=self.base) as temporary:
            workdir = Path(temporary)
            try:
                torch.cuda.synchronize(device)
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                decision = measure(snapshot, key, info, workdir)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                problem = exc
                feasible = [
                    read_json(path.with_name(path.name.removesuffix(".result.json") + ".job"))[
                        "microbatch"
                    ]
                    for path in workdir.glob("*.result.json")
                    if read_json(path).get("status") == "ok"
                ]
                microbatch = min(feasible, default=baseline_microbatches(len(batch[1]))[-1])
                decision = {"status": "retryable-fallback", "microbatch": microbatch}
            finally:
                after = {name: state_hash(value) for name, value in self.live_state(device).items()}
                if before != after:
                    raise RuntimeError(
                        "Calibration changed live model, optimizer, gradients or RNG"
                    )

        if problem is not None:
            warnings.warn(
                f"Effective-update calibration interrupted: {problem}", RuntimeWarning, stacklevel=2
            )
            self.event(
                key, decision["status"], microbatch=decision["microbatch"], error=str(problem)
            )
            return decision
        if not valid_receipt(decision, key):
            raise RuntimeError("Incomplete effective-update calibration evidence")
        decision.update(
            state_before=before, state_after=after, fixed_input_digest=state_hash(snapshot["batch"])
        )
        receipt = self.base / (key + ".json")
        atomic_json(receipt, decision)
        self.event(
            key,
            "calibrated",
            decision=decision["status"],
            microbatch=decision["microbatch"],
            receipt=str(receipt),
            state_preserved=True,
        )
        return decision

    def logical_update(self, batch: Batch) -> dict[str, Any]:
        count = len(batch[1])
        target = self.recipe["effective_batch_size"]
        if count == target and batch[0].is_cuda:
            _, decision = self.configure(batch)
            microbatch = decision["microbatch"]
        else:
            # Prefer a previously validated size on shorter updates.
            microbatch = min(
                count, min((d["microbatch"] for d in self.decisions.values()), default=32)
            )
        options = {"compiled_update": self.compiled_update} if self.compiled_update else {}
        return run_update(
            self.model,
            self.runtime_for(microbatch),
            self.optimizer,
            batch,
            microbatch,
            self.recipe["precision"],
            metrics=True,
            **options,
        )
