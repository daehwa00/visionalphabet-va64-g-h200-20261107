"""Automatic microbatch and kernel selection for one logical training update."""

from __future__ import annotations

import fcntl
import hashlib
import os
import socket
import tempfile
import warnings
from pathlib import Path

import torch

from automatic_training_layout import (
    AutomaticTrainingRuntime,
    atomic_json,
    cpu_tree,
    digest,
    read_json,
    state_hash,
    workload_geometries,
)
from effective_batch_search import measure, valid_receipt
from effective_batch_update import run_update

PACKAGED = Path(__file__).resolve().parents[1] / "src/lnet/validated_effective_updates.json"


class EffectiveTrainingRuntime(AutomaticTrainingRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = self.base.parent / "effective-updates"
        self.audit = self.audit.parent / "effective-update-calibration"
        self.audit.mkdir(parents=True, exist_ok=True)
        files = (
            "effective_batch_runtime.py",
            "effective_batch_search.py",
            "effective_batch_proposals.py",
            "effective_batch_update.py",
            "effective_batch_epoch.py",
            "imagenet100_data_runtime.py",
            "tune_path_backward.py",
        )
        self.policy_revision = hashlib.sha256(
            b"".join(Path(__file__).with_name(s).read_bytes() for s in files)
        ).hexdigest()

    def saved(self, key):
        for path, origin in ((self.base / (key + ".json"), "local"), (PACKAGED, "bundled")):
            doc = read_json(path)
            if origin == "bundled":
                doc = doc.get(key, {})
            if valid_receipt(doc, key):
                self.event(
                    key,
                    "reused",
                    source=origin,
                    decision=doc["status"],
                    microbatch=doc["microbatch"],
                )
                return doc
        return None

    def configure(self, batch):
        x, y, _, mix = batch
        device = x.device
        props = torch.cuda.get_device_properties(device)
        phase = "warmup" if self.epoch_index < self.recipe.get("warmup_epochs", 0) else "scheduled"
        zeros = tuple(float(g["lr"]) == 0 for g in self.optimizer.param_groups)
        memory_fraction = torch.cuda.get_per_process_memory_fraction(device)
        free, total = torch.cuda.mem_get_info(device)
        available = min(free + torch.cuda.memory_reserved(device), total * memory_fraction)
        # Ignore ordinary driver bookkeeping on an otherwise empty card. Distinguish
        # co-tenancy and allocator limits without keying on each transient allocation.
        budget_gib = (
            int(total * memory_fraction // 2**30)
            if available >= 0.95 * total * memory_fraction
            else int(available // 2**30)
        )
        signature = (
            budget_gib,
            tuple(x.shape),
            str(x.dtype),
            tuple(x.stride()),
            torch.is_tensor(mix),
            phase,
            zeros,
        )
        if signature in self.keys:
            key = self.keys[signature]
            return key, self.decisions[key]
        info = {
            **self.identity,
            "policy": "effective-update-v2",
            "policy_revision": self.policy_revision,
            "input_shape": list(x.shape),
            "input_dtype": str(x.dtype),
            "input_stride": list(x.stride()),
            "target_dtype": str(y.dtype),
            "per_sample_mix": torch.is_tensor(mix),
            "effective_samples": len(y),
            "loader_batch": self.recipe["batch_size"],
            "schedule_phase": phase,
            "zero_lr_groups": zeros,
            "hardware_memory_bytes": props.total_memory,
            "multiprocessors": props.multi_processor_count,
            "memory_fraction": memory_fraction,
            "memory_budget_gib": budget_gib,
            "device_uuid": str(getattr(props, "uuid", props.name)),
        }
        key = digest(info)
        decision = self.saved(key)
        if decision is None:
            self.base.mkdir(parents=True, exist_ok=True)
            lock_name = "gpu-" + digest((socket.gethostname(), info["device_uuid"])) + ".lock"
            with (
                (self.base / (key + ".lock")).open("a+") as job,
                (self.base / lock_name).open("a+") as card,
            ):
                fcntl.flock(job, fcntl.LOCK_EX)
                fcntl.flock(card, fcntl.LOCK_EX)
                decision = self.saved(key)
                if decision is None:
                    self.event(key, "calibrating", effective_samples=len(y))
                    live = self.live_state(device)
                    before = {name: state_hash(value) for name, value in live.items()}
                    snapshot = dict(
                        variant=self.variant,
                        model_config=self.model_config,
                        factories=self.factories,
                        recipe=self.recipe,
                        settings=self.settings,
                        model=cpu_tree(live["model"]),
                        optimizer=cpu_tree(live["optimizer"]),
                        rng=cpu_tree(live["rng"]),
                        batch=cpu_tree(batch),
                        index=device.index
                        if device.index is not None
                        else torch.cuda.current_device(),
                    )
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                    with tempfile.TemporaryDirectory(
                        prefix=".calibration-", dir=self.base
                    ) as temporary:
                        problem = None
                        try:
                            decision = measure(snapshot, key, info, Path(temporary))
                        except (OSError, RuntimeError) as exc:
                            problem = exc
                            feasible = []
                            for path in Path(temporary).glob("*.result.json"):
                                result = read_json(path)
                                if result.get("status") == "ok":
                                    job = read_json(path.with_suffix("").with_suffix(".job"))
                                    if job.get("microbatch") and not job.get("records"):
                                        feasible.append(job["microbatch"])
                            decision = {"microbatch": min(feasible, default=32), "records": {}}
                        finally:
                            after = {
                                name: state_hash(value)
                                for name, value in self.live_state(device).items()
                            }
                            if before != after:
                                raise RuntimeError(
                                    "Calibration changed the live model, optimizer, gradients or RNG"
                                )
                    if problem is not None:
                        warnings.warn(
                            f"Effective-update calibration interrupted: {problem}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self.event(
                            key,
                            "retryable-fallback",
                            microbatch=decision["microbatch"],
                            error=str(problem),
                            state_preserved=True,
                        )
                    else:
                        if not valid_receipt(decision, key):
                            raise RuntimeError("Incomplete effective-update calibration evidence")
                        decision.update(
                            state_before=before,
                            state_after=after,
                            fixed_input_digest=state_hash(snapshot["batch"]),
                        )
                        atomic_json(self.base / (key + ".json"), decision)
                        self.event(
                            key,
                            "calibrated",
                            decision=decision["status"],
                            microbatch=decision["microbatch"],
                            receipt=str(self.base / (key + ".json")),
                            state_preserved=True,
                        )
        self.keys[signature] = key
        self.decisions[key] = decision
        return key, decision

    def logical_update(self, batch):
        if len(batch[1]) == 1024:
            key, decision = self.configure(batch)
            micro, records = decision["microbatch"], decision["records"]
        else:
            # A short final group has a different batch-dependent forward and no full-update receipt.
            key, micro, records = "tail-" + str(len(batch[1])), min(128, len(batch[1])), {}
        with workload_geometries(records):
            return run_update(
                self.model,
                self.runtime_for(key, records),
                self.optimizer,
                batch,
                micro,
                self.recipe["precision"],
                metrics=True,
            )
