# ruff: noqa: ANN001, ANN202, C901, PLC0415, PLR0915, SLF001
"""Data loading, CPU placement, and device prefetch for ImageNet-100."""

from __future__ import annotations

# DataLoader batches and torchvision transforms expose dynamic tuple types.
# pyright: reportArgumentType=false, reportExplicitAny=false
import hashlib
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from alphabet.training_compiler import training_compiler_features

if TYPE_CHECKING:
    from collections.abc import Iterator


CUDA_GRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})
PREFETCH_FACTOR = 2
_PREFETCH_STREAMS: dict[int, torch.cuda.Stream] = {}
WANDB_MODEL_ALIASES = {"product_four": "Product4", "pole_free": "PoleFree"}


def begin_cudagraph_step(device: torch.device) -> None:
    """Declare one model invocation boundary for compiled CUDA Graph runtimes."""
    if device.type != "cuda":
        return
    if "cuda_graphs" in training_compiler_features() or os.environ.get(
        "LNET_CUDAGRAPHS_ACTIVE",
        str(int(os.environ.get("LNET_COMPILE_MODE") in CUDA_GRAPH_COMPILE_MODES)),
    ) == "1":
        mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark is not None:
            mark()


def active_loader_workers(requested: int) -> int:
    workers = int(os.environ.get("LNET_DATALOADER_WORKERS", requested))
    if workers < 0:
        message = "DataLoader workers must be nonnegative"
        raise ValueError(message)
    return workers


def persistent_loader_workers(active_workers: int) -> bool:
    return active_workers > 0 and os.environ.get("LNET_PERSISTENT_WORKERS", "1") == "1"


def parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for field in value.split(","):
        bounds = field.strip().split("-", maxsplit=1)
        if not bounds[0]:
            message = f"invalid CPU affinity: {value}"
            raise ValueError(message)
        start = int(bounds[0])
        stop = int(bounds[-1])
        if min(start, stop) < 0 or stop < start:
            message = f"invalid CPU affinity: {value}"
            raise ValueError(message)
        cpus.update(range(start, stop + 1))
    if not cpus:
        message = "CPU affinity cannot be empty"
        raise ValueError(message)
    return cpus


def format_cpu_set(cpus: set[int]) -> str:
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def cpu_topology(cpus: set[int]) -> dict[tuple[int, int], set[int]]:
    topology: dict[tuple[int, int], set[int]] = defaultdict(set)
    root = Path("/sys/devices/system/cpu")
    for cpu in sorted(cpus):
        topology_root = root / f"cpu{cpu}" / "topology"
        package = int((topology_root / "physical_package_id").read_text().strip())
        core = int((topology_root / "core_id").read_text().strip())
        topology[(package, core)].add(cpu)
    return dict(topology)


def partition_physical_cores(
    topology: dict[tuple[int, int], set[int]],
    *,
    partition: int,
    partitions: int,
) -> set[int]:
    if partitions < 1 or partition not in range(partitions):
        message = "CPU partition index is outside the available GPU partitions"
        raise ValueError(message)
    cores = [siblings for _, siblings in sorted(topology.items())]
    quotient, remainder = divmod(len(cores), partitions)
    start = partition * quotient + min(partition, remainder)
    count = quotient + (partition < remainder)
    return set().union(*cores[start : start + count]) if count else set()


def configure_cpu_affinity() -> set[int]:
    """Give a single-GPU run a stable, disjoint physical-core partition."""
    allowed = set(os.sched_getaffinity(0))
    explicit = os.environ.get("LNET_CPU_AFFINITY")
    if explicit:
        selected = parse_cpu_set(explicit)
        if not selected <= allowed:
            message = "requested CPU affinity falls outside the launcher affinity"
            raise RuntimeError(message)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_paths = sorted(Path("/proc/driver/nvidia/gpus").glob("*"))
        if not visible.isdecimal() or len(gpu_paths) < 2:
            selected = allowed
        else:
            online = set(range(os.cpu_count() or 1))
            if allowed != online:
                selected = allowed
            else:
                selected = partition_physical_cores(
                    cpu_topology(allowed),
                    partition=int(visible),
                    partitions=len(gpu_paths),
                )
    if not selected:
        message = "automatic GPU CPU partition is empty"
        raise RuntimeError(message)
    os.sched_setaffinity(0, selected)
    os.environ["LNET_CPU_AFFINITY_ACTIVE"] = format_cpu_set(selected)
    return selected


def compact_wandb_name(variant: str) -> str:
    """Return a short model-only label; provenance stays in W&B config."""
    if variant in WANDB_MODEL_ALIASES:
        return WANDB_MODEL_ALIASES[variant]
    name = variant
    for prefix in ("capacity_dual_", "alphabet2d_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    replacements = (
        ("double_precomplex_fc", "DoublePreFC"),
        ("precomplex_fc", "PreFC"),
        ("first_order", "FO"),
        ("cccn_shortcut", "CCCN"),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name.replace("_", "-")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = (
        path
        for split in ("train", "val")
        for directory in sorted((root / split).glob("*"))
        for path in sorted(directory.glob("*"))
        if path.is_file()
    )
    train_count = 0
    validation_count = 0
    for path in files:
        relative = path.relative_to(root)
        if relative.parts[0] == "train":
            train_count += 1
        else:
            validation_count += 1
        digest.update(str(relative).encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest(), train_count, validation_count


def transforms_for_imagenet100(
    *, randaugment_magnitude=9, randaugment_num_ops=2, random_erasing_prob=0.25
) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    standard_deviation = (0.229, 0.224, 0.225)
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
            transforms.RandomErasing(p=random_erasing_prob),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    return train, evaluation


def init_loader_worker(_worker_id: int) -> None:
    torch.set_num_threads(1)


def clean_worker_context():
    # The server is a fresh Python process and never runs CUDA or compiler work.
    multiprocessing.set_forkserver_preload(["imagenet100_worker_preload"])
    return multiprocessing.get_context("forkserver")


def loaders(
    data_root: Path,
    *,
    batch_size: int,
    validation_batch_size: int | None = None,
    workers: int,
    training_generator: torch.Generator,
    val_persistent_workers: bool = False,
    isolated_validation: bool = False,
    prefetch_factor: int | None = PREFETCH_FACTOR,
    fast_gpu_augment: bool = False,
    train_persistent_workers: bool = True,
    randaugment_magnitude: int = 9,
    randaugment_num_ops: int = 2,
    color_jitter: float = 0.0,
    random_erasing_prob: float = 0.25,
    compact_manifest: bool = False,
    gpu_randaugment: bool = False,
    tensor_crop: bool = False,
    clean_loader_workers: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    if tensor_crop and (not fast_gpu_augment or (randaugment_num_ops and not gpu_randaugment)):
        raise ValueError(
            "Tensor crop requires fast augmentation and either GPU RandAugment or no RandAugment"
        )
    if color_jitter and not fast_gpu_augment:
        raise ValueError("ColorJitter requires fast GPU augmentation")
    train_transform, evaluation_transform = transforms_for_imagenet100(
        randaugment_magnitude=randaugment_magnitude,
        randaugment_num_ops=randaugment_num_ops,
        random_erasing_prob=random_erasing_prob,
    )
    if compact_manifest:
        from imagenet_compact_manifest import CompactImageFolder
        from torchvision.datasets.folder import pil_loader

        dataset = CompactImageFolder
        default_loader = {"loader": pil_loader}
        dataset_options = {"mapped": clean_loader_workers}
    else:
        dataset = datasets.ImageFolder
        default_loader = {}
        dataset_options = {}
    train_options = {}
    if fast_gpu_augment:
        from imagenet100_fast_augment import Uint8TrainingTransform, collate_uint8, load_rgb

        crop = train_transform.transforms[0]
        if tensor_crop:
            from imagenet100_fast_augment import load_rgb_tensor
            from imagenet_uint8_crop import Uint8BicubicCropAndFlip

            uint8_transform = Uint8BicubicCropAndFlip(crop)
            loader = load_rgb_tensor
        else:
            randaugment = (
                train_transform.transforms[2]
                if randaugment_num_ops and not gpu_randaugment
                else None
            )
            uint8_transform = Uint8TrainingTransform(crop, randaugment)
            loader = load_rgb
        train_options["collate_fn"] = collate_uint8
        train_dataset = dataset(
            data_root / "train", uint8_transform, loader=loader, **dataset_options
        )
    else:
        train_dataset = dataset(
            data_root / "train", train_transform, **default_loader, **dataset_options
        )
    validation_dataset = dataset(
        data_root / "val", evaluation_transform, **default_loader, **dataset_options
    )
    if train_dataset.classes != validation_dataset.classes:
        message = "ImageNet-100 train and validation class sets differ"
        raise RuntimeError(message)
    active_workers = active_loader_workers(workers)
    persistent_workers = train_persistent_workers and persistent_loader_workers(active_workers)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": active_workers,
        "pin_memory": True,
    }
    if active_workers > 0:
        common["worker_init_fn"] = init_loader_worker
        if clean_loader_workers:
            common["multiprocessing_context"] = clean_worker_context()
        common["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=training_generator,
        drop_last=True,
        persistent_workers=persistent_workers,
        **train_options,
        **common,
    )
    train_loader.random_erasing_prob = random_erasing_prob
    train_loader.gpu_randaugment = gpu_randaugment and randaugment_num_ops > 0
    train_loader.color_jitter = color_jitter
    train_loader.randaugment_magnitude = randaugment_magnitude
    validation_workers = min(4, active_workers) if isolated_validation else active_workers
    validation_options = {**common, "num_workers": validation_workers}
    if validation_workers == 0:
        validation_options.pop("prefetch_factor", None)
        validation_options.pop("multiprocessing_context", None)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        persistent_workers=val_persistent_workers and validation_workers > 0,
        generator=torch.Generator().manual_seed(10_502) if isolated_validation else None,
        **{**validation_options, "batch_size": validation_batch_size or batch_size},
    )
    return train_loader, validation_loader


def close_persistent_loader(loader: DataLoader[Any]) -> None:
    # PyTorch has no public explicit shutdown for a persistent iterator.
    iterator = loader._iterator
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def move_batch(
    batch_inputs: Tensor,
    batch_targets: Tensor,
    device: torch.device,
    *,
    channels_last: bool,
) -> tuple[Tensor, Tensor]:
    memory_format = torch.channels_last if channels_last else torch.preserve_format
    inputs = batch_inputs.to(
        device=device,
        non_blocking=True,
        memory_format=memory_format,
    )
    targets = batch_targets.to(device=device, non_blocking=True)
    return inputs, targets


def device_batches(
    loader: DataLoader[Any],
    device: torch.device,
    *,
    channels_last: bool,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Transfer batches in order while overlapping the next H2D copy on CUDA."""
    if device.type != "cuda":
        for batch_inputs, batch_targets in loader:
            yield move_batch(
                batch_inputs,
                batch_targets,
                device,
                channels_last=channels_last,
            )
        return

    from imagenet100_fast_augment import CudaTrainingAugment, collate_uint8

    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index not in _PREFETCH_STREAMS:
        _PREFETCH_STREAMS[device_index] = torch.cuda.Stream(device=device_index)
    prefetch_stream = _PREFETCH_STREAMS[device_index]
    iterator = iter(loader)
    fast = getattr(loader, "collate_fn", None) is collate_uint8
    gpu_randaugment = getattr(loader, "gpu_randaugment", False)
    with torch.cuda.stream(prefetch_stream):
        augment = (
            CudaTrainingAugment(
                device,
                channels_last=channels_last,
                random_erasing_prob=getattr(loader, "random_erasing_prob", 0.25),
                color_jitter_strength=getattr(loader, "color_jitter", 0.0),
            )
            if fast
            else None
        )

    if gpu_randaugment:
        from imagenet_gpu_randaugment import GpuRandAugment

        with torch.cuda.stream(prefetch_stream):
            randaugment = GpuRandAugment(device, magnitude=loader.randaugment_magnitude)

    def stage(batch_inputs: Tensor, batch_targets: Tensor) -> tuple[Tensor, Tensor]:
        with torch.cuda.stream(prefetch_stream):
            inputs, targets = move_batch(batch_inputs, batch_targets, device, channels_last=False)
            if gpu_randaugment:
                inputs = randaugment(inputs)
            if augment is not None:
                inputs, targets = augment(inputs, targets)
            return inputs, targets

    try:
        batch_inputs, batch_targets = next(iterator)
    except StopIteration:
        return
    next_inputs, next_targets = stage(batch_inputs, batch_targets)

    while True:
        current_stream = torch.cuda.current_stream(device=device)
        current_stream.wait_stream(prefetch_stream)
        inputs, targets = next_inputs, next_targets
        inputs.record_stream(current_stream)
        targets.record_stream(current_stream)

        try:
            batch_inputs, batch_targets = next(iterator)
        except StopIteration:
            has_next = False
        else:
            has_next = True
            next_inputs, next_targets = stage(batch_inputs, batch_targets)

        if channels_last and not fast:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        yield inputs, targets
        if not has_next:
            break


__all__ = [
    "PREFETCH_FACTOR",
    "active_loader_workers",
    "begin_cudagraph_step",
    "compact_wandb_name",
    "configure_cpu_affinity",
    "dataset_digest",
    "device_batches",
    "digest",
    "loaders",
    "persistent_loader_workers",
    "transforms_for_imagenet100",
]
