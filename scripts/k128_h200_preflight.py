"""Choose one direct H200 batch size for both K128 experiment arms."""

# Standalone sibling scripts share internal helpers; timing/contract JSON is dynamic.
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false, reportExplicitAny=false

from __future__ import annotations

import gc
import json
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imagenet1k_runtime as runtime
import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

EXECUTION_VERSION = "k128-h200-fast-v1"
BATCHES = (256, 512)


def fast_execution() -> runtime.ExecutionOptions:
    return runtime.ExecutionOptions(
        compile_model=True,
        gpu_mixup=True,
        non_blocking_transfer=True,
        prefetch_factor=2,
        metric_sync_interval=100,
    )


def saved_selection(root: Path) -> dict[str, Any] | None:
    """A resumed fast campaign keeps its chosen batch, including on a new pod."""
    selected = None
    for variant in ("baseline", "improved"):
        path = root / variant / "contracts" / "execution-selection.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("execution_version") != EXECUTION_VERSION
            or value.get("selected_batch") not in BATCHES
            or value.get("effective_batch") != value.get("selected_batch")
            or value.get("gradient_accumulation_steps") != 1
        ):
            raise RuntimeError("Saved K128 batch selection is incompatible")
        if selected is not None and value != selected:
            raise RuntimeError("The two arms have conflicting batch selections")
        selected = value
    return selected


def choose_batch(probes: dict[int, dict[str, Any]]) -> int:
    """Prefer 512 when its median throughput exceeds 256 by more than noise."""
    small = probes[256]
    large = probes[512]
    if small.get("status") != "passed":
        raise RuntimeError("The batch-256 preflight must pass")
    if large.get("status") == "oom":
        return 256
    if large.get("status") != "passed":
        raise RuntimeError("The batch-512 preflight did not complete")
    return 512 if large["images_per_second"] >= small["images_per_second"] * 1.03 else 256


def _probe_batch(
    batch: int,
    root: Path,
    builder: Callable[[str], nn.Module],
    optimizer_builder: Callable[[nn.Module, float, torch.device], torch.optim.Optimizer],
    campaign_id: str,
    data_root: Path,
) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(501)
    model = builder("baseline").to(device)
    model = model.to(memory_format=torch.channels_last).train()  # pyright: ignore[reportCallIssue]
    optimizer = optimizer_builder(model, 0.003, device)
    runtime._compile_model_in_place(model, fast_execution())
    images = torch.randn(batch, 3, 224, 224, device=device).contiguous(
        memory_format=torch.channels_last
    )
    targets = torch.arange(batch, device=device) % 1000

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        loss.backward()
        optimizer.step()
        return loss

    started = time.monotonic()
    loss: torch.Tensor | None = None
    print(f"K128_BATCH_PROBE_START batch={batch}", flush=True)
    for index in range(4):
        loss = step()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite batch-{batch} preflight loss")
        runtime._atomic_json(
            root / "current.json",
            {
                "campaign_id": campaign_id,
                "job": "k128-baseline-501",
                "stage": "preflight",
                "probe_step": batch * 100 + index,
                "output": str(root / "baseline"),
            },
        )
    warmup_seconds = time.monotonic() - started
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(9):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    if loss is None or not bool(torch.isfinite(loss)) or not all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    ):
        raise RuntimeError(f"Nonfinite batch-{batch} loss or gradient")
    median = statistics.median(times)
    result = {
        "status": "passed",
        "physical_batch": batch,
        "effective_batch": batch,
        "gradient_accumulation_steps": 1,
        "images_per_second": batch / median,
        "median_step_ms": median * 1000,
        "times_ms": [value * 1000 for value in times],
        "warmup_seconds": warmup_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "loss": float(loss.detach()),
    }
    # Decide on the real training path: larger GPU-only batches need not improve
    # throughput when decoding or augmentation is the limiting factor.
    task = runtime.BaselineTask(
        phase="full", model_key="k128_baseline", seed=501, learning_rate=0.003,
        epochs=200, data_root=data_root, output_dir=root / "preflight" / str(batch),
        result_path=root / "preflight" / f"{batch}.json",
        checkpoint_path=root / "preflight" / f"{batch}.pt",
        batch_size=batch, effective_batch_size=batch, workers=8, execution=fast_execution(),
    )
    loaders = runtime._build_loaders(task, device)
    if (
        len(cast("ImageFolder", loaders.train.dataset)),
        len(cast("ImageFolder", loaders.validation.dataset)),
    ) != (1_281_167, 50_000):
        raise RuntimeError("Batch selection requires the full ImageNet-1K dataset")
    measured = _TimedBatches(loaders.train, warmup=8, measured=32)
    try:
        runtime._train_epoch(
            # The bounded wrapper implements the len/iteration interface used by the loop.
            model, cast("DataLoader[Any]", cast("object", measured)), optimizer,
            runtime._make_mixup(task), task, device, 0
        )
        torch.cuda.synchronize()
        if measured.started is None:
            raise RuntimeError("Training throughput timer did not start")
        elapsed = time.perf_counter() - measured.started
    finally:
        measured.close()
    result["gpu_images_per_second"] = result["images_per_second"]
    result["gpu_median_step_ms"] = result.pop("median_step_ms")
    result["gpu_times_ms"] = result.pop("times_ms")
    result["images_per_second"] = 32 * batch / elapsed
    result["pipeline_seconds"] = elapsed
    result["pipeline_mean_step_ms"] = elapsed * 1000 / 32
    result["pipeline_measured_images"] = 32 * batch
    result["pipeline_warmup_batches"] = 8
    print("K128_BATCH_PROBE_JSON=" + json.dumps(result), flush=True)
    return result


class _TimedBatches:
    """Bound the loader without including process startup/shutdown in timing."""

    def __init__(self, loader: Any, *, warmup: int, measured: int) -> None:
        self.loader = loader
        self.warmup = warmup
        self.measured = measured
        self.started: float | None = None
        self.iterator: Any = None

    def __len__(self) -> int:
        return self.warmup + self.measured

    def __iter__(self) -> Iterator[Any]:
        self.iterator = iter(self.loader)
        for index in range(len(self)):
            if index == self.warmup:
                torch.cuda.synchronize()
                self.started = time.perf_counter()
            batch = next(self.iterator)
            yield batch

    def close(self) -> None:
        # The bounded iterator retains prefetched batches beyond our sample.
        # Release its owned worker processes after the timed region.
        shutdown = getattr(self.iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        self.iterator = None


def select_batch(
    root: Path,
    builder: Callable[[str], nn.Module],
    optimizer_builder: Callable[[nn.Module, float, torch.device], torch.optim.Optimizer],
    source: dict[str, str],
    data_root: Path,
    campaign_id: str = "k128-optimization-20260929",
) -> dict[str, Any]:
    selected = saved_selection(root)
    if selected is None:
        probes = {}
        for batch in BATCHES:
            try:
                probes[batch] = _probe_batch(
                    batch, root, builder, optimizer_builder, campaign_id, data_root
                )
            except torch.OutOfMemoryError:
                if batch == 256:
                    raise
                probes[batch] = {"status": "oom", "physical_batch": batch}
                print("K128_BATCH_PROBE_OOM batch=512", flush=True)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        chosen = choose_batch(probes)
        selected = {
            "campaign_id": campaign_id,
            "execution_version": EXECUTION_VERSION,
            "selected_batch": chosen,
            "effective_batch": chosen,
            "gradient_accumulation_steps": 1,
            "selection_rule": "512 if >=1.03x batch-256 steady-state training throughput; otherwise 256",
            "gpu": torch.cuda.get_device_name(),
            "source": source,
            "probes": {str(key): value for key, value in probes.items()},
            "measurement": "ImageNet loading, augmentation, H2D, model and optimizer; compilation and loader warmup excluded",
        }
    for variant in ("baseline", "improved"):
        runtime._atomic_json(
            root / variant / "contracts" / "execution-selection.json", selected
        )
    runtime._atomic_json(root / "batch-selection.json", selected)
    print("K128_BATCH_SELECTION_JSON=" + json.dumps(selected), flush=True)
    return selected
