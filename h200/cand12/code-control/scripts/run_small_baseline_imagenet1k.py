"""Train official compact baselines with the retained VA ImageNet-1K recipe."""

# The internal runtime intentionally exposes its serialization/loader helpers here.
# pyright: reportPrivateUsage=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import random
import shutil
import signal
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imagenet1k_runtime as runtime
import numpy as np
import small_baseline_models as models
import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Match timm Mixup's global NumPy stream used by the VA runs.
# ruff: noqa: NPY002
MODELS = ("efficientvmamba_t", "mobilenetv4_conv_small", "repvit_m0_9")
SEEDS = (0, 501, 509, 521)


def build_model(root: Path, model_name: str) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError("unsupported small baseline model")
    return models.build_model(root, model_name)


def extension_binding(model_name: str) -> dict[str, str]:
    if model_name != "efficientvmamba_t":
        return {}
    extension = importlib.import_module("selective_scan_cuda_core")
    if extension.__file__ is None:
        raise ValueError("EfficientVMamba extension has no binary file")
    path = Path(extension.__file__).resolve()
    if path.suffix != ".so" or not path.is_file():
        raise ValueError("EfficientVMamba requires the compiled selective scan extension")
    return {"selective_scan_cuda_core_binary": hashlib.sha256(path.read_bytes()).hexdigest()}


def make_task(args: argparse.Namespace) -> runtime.BaselineTask:
    binding = models.source_binding(args.models_root, args.source_manifest)
    binding.update(extension_binding(args.model))
    source_manifest = json.loads(args.source_manifest.read_text())
    code_root = Path(__file__).resolve().parent.parent
    files = source_manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest must bind code files")
    for name, expected in files.items():
        path = (code_root / name).resolve()
        if not path.is_relative_to(code_root) or not path.is_file():
            raise ValueError("source manifest code path is invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source manifest code hash mismatch: {name}")
    manifest = json.loads(args.data_manifest.read_text())
    if manifest["root"] != str(args.data_root.resolve()) or manifest["counts"] != {
        "train": 1281167,
        "val": 50000,
    }:
        raise ValueError("ImageNet manifest does not match the full dataset")
    output = args.output_root.resolve() / args.model / f"seed_{args.seed}"
    return runtime.BaselineTask(
        phase="full",
        model_key=args.model,
        seed=args.seed,
        learning_rate=0.003,
        epochs=100,
        data_root=args.data_root.resolve(),
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=output / "checkpoint.pt",
        source_root=args.models_root.resolve(),
        batch_size=args.batch_size,
        workers=args.workers,
        resume=args.resume,
        execution=runtime.ExecutionOptions(
            compile_model=not args.no_compile,
            gpu_mixup=True,
            non_blocking_transfer=True,
            prefetch_factor=2,
            metric_sync_interval=100,
        ),
        source_sha256={
            **binding,
            "source_manifest": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
            "model_adapter": hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest(),
            "entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_manifest": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
        },
    )


def parameter_inventory(model: nn.Module) -> dict[str, int]:
    return {
        "stored_elements": sum(p.numel() for p in model.parameters()),
        "trainable_elements": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stored_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }


def reset_data_rng(seed: int) -> None:
    random.seed(seed + 1_000_000)
    np.random.seed(seed + 1_000_000)
    torch.manual_seed(seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1_000_000)


def make_builder(
    args: argparse.Namespace, task: runtime.BaselineTask
) -> Callable[[str, str | Path | None, int], nn.Module]:
    def builder(key: str, _source: str | Path | None, classes: int) -> nn.Module:
        if key != task.model_key or classes != 1000:
            raise ValueError("builder/task mismatch")
        model = build_model(args.models_root, args.model)
        runtime._atomic_json(
            task.output_dir / "parameter-inventory.json", parameter_inventory(model)
        )
        drop_paths = sorted(
            {
                float(value)
                for m in model.modules()
                if isinstance(value := getattr(m, "drop_prob", None), (int, float))
            }
        )
        dropouts = sorted({float(m.p) for m in model.modules() if isinstance(m, nn.Dropout)})
        runtime._atomic_json(
            task.output_dir / "model-definition.json",
            {
                "model": args.model,
                "source_manifest": str(args.source_manifest.resolve()),
                "source_manifest_sha256": task.source_sha256["source_manifest"],
                "pretrained": False,
                "drop_path_probabilities": drop_paths,
                "dropout_probabilities": dropouts,
                "teacher_distillation": False,
                "note": "Official public model implementation and default architectural stochastic "
                "depth are retained; ImageNet training uses the shared VA recipe.",
            },
        )
        reset_data_rng(task.seed)
        return model

    return builder


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.stop_file.is_file():
        raise runtime.TrainingInterrupted("stop requested before model/data initialization")
    task = make_task(args)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_updates:
        return smoke(args, task)
    started = time.monotonic()
    state: dict[str, Any] = {}
    best_path = task.output_dir / "best.json"
    best = json.loads(best_path.read_text())["accuracy"] if best_path.exists() else -math.inf

    def optimizer_builder(
        model: nn.Module, lr: float, device: torch.device
    ) -> torch.optim.Optimizer:
        optimizer = runtime._build_optimizer(model, lr, device)
        updates = 0

        def before_step(
            opt: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
        ) -> None:
            nonlocal updates
            updates += 1
            if updates % 100 == 0:
                norms = [
                    p.grad.detach().float().norm()
                    for group in opt.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                norm = torch.stack(norms).norm().item()
                if not math.isfinite(norm):
                    raise RuntimeError("nonfinite gradient norm")
                state["gradient_norm"] = norm

        optimizer.register_step_pre_hook(before_step)
        return optimizer

    def step_callback(row: dict[str, Any]) -> None:
        if not math.isfinite(row["loss"]):
            raise RuntimeError("nonfinite training loss")
        record = {**row, **state, "elapsed_seconds": time.monotonic() - started, **memory_metrics()}
        with (task.output_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("BASELINE_TRAIN_STEP=" + json.dumps(record), flush=True)

    def epoch_callback(_model: nn.Module, row: dict[str, Any]) -> None:
        nonlocal best
        for group in ("train", "validation"):
            if any(not math.isfinite(float(value)) for value in row[group].values()):
                raise RuntimeError(f"nonfinite {group} epoch metrics")
        accuracy = row["validation"]["accuracy"]
        if accuracy > best:
            temporary = task.output_dir / ".best.pt.tmp"
            shutil.copyfile(task.checkpoint_path, temporary)
            temporary.replace(task.output_dir / "best.pt")
            best = accuracy
            runtime._atomic_json(best_path, {"epoch": row["epoch"], "accuracy": best})
        runtime._atomic_json(task.output_dir / "latest-epoch.json", row)
        # Rewrite derived history so recovery cannot duplicate the last epoch.
        checkpoint = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
        temporary = task.output_dir / ".epochs.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r) + "\n" for r in checkpoint["history"]))
        temporary.replace(task.output_dir / "epochs.jsonl")
        print("BASELINE_EPOCH=" + json.dumps(row), flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = runtime.run_task(
        task,
        model_builder=make_builder(args, task),
        loader_builder=runtime._build_loaders,
        optimizer_builder=optimizer_builder,
        epoch_callback=epoch_callback,
        step_callback=step_callback,
        stop_requested=args.stop_file.is_file,
        stop_without_checkpoint=True,
    )
    runtime._atomic_json(
        task.output_dir / "cost.json",
        {
            "model": args.model,
            "seed": args.seed,
            "status": result["status"],
            "process_seconds": time.monotonic() - started,
            **memory_metrics(),
            "training_seconds": result.get("training_seconds"),
        },
    )
    return result


def memory_metrics() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def edge_smoke(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bundle: runtime.LoaderBundle,
    task: runtime.BaselineTask,
    device: torch.device,
    full_train_batch: Any,
    stop_requested: Callable[[], bool],
) -> dict[str, Any]:
    """Exercise actual validation and the full-loader remainder batch shapes."""
    from torch.utils.data import DataLoader, Subset

    train_size = len(bundle.train.dataset)  # pyright: ignore[reportArgumentType]
    validation_size = len(bundle.validation.dataset)  # pyright: ignore[reportArgumentType]
    train_tail_size = train_size % task.batch_size
    validation_tail_size = validation_size % task.batch_size
    if not train_tail_size or not validation_tail_size:
        raise ValueError("smoke expects ImageNet's nonempty training/validation remainder")
    train_tail = next(
        iter(
            DataLoader(
                Subset(bundle.train.dataset, range(train_size - train_tail_size, train_size)),
                batch_size=task.batch_size,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
        )
    )
    train_batches = [full_train_batch, train_tail]

    class TailBatches:
        def __len__(self) -> int:
            return len(train_batches)

        def __iter__(self) -> Iterator[Any]:
            return iter(tuple(tuple(t.clone() for t in batch) for batch in train_batches))

    train_metrics, updates = runtime._train_epoch(
        model,
        cast("DataLoader[Any]", cast("object", TailBatches())),
        optimizer,
        runtime._make_mixup(task),
        task,
        device,
        0,
        stop_requested=stop_requested,
    )
    expected_examples = task.batch_size + train_tail_size
    if train_metrics["examples"] != expected_examples or not math.isfinite(train_metrics["loss"]):
        raise RuntimeError("full-plus-tail training smoke failed")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("nonfinite parameters after remainder training smoke")
    validation_indices = [
        *range(task.batch_size),
        *range(validation_size - validation_tail_size, validation_size),
    ]
    validation_loader = DataLoader(
        Subset(bundle.validation.dataset, validation_indices),
        batch_size=task.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation = runtime._evaluate(
        model,
        validation_loader,
        task,
        device,
        stop_requested=stop_requested,
    )
    if validation["examples"] != task.batch_size + validation_tail_size:
        raise RuntimeError("validation full/tail smoke example count differs")
    return {
        "train_full_batch": task.batch_size,
        "train_tail_batch": train_tail_size,
        "train_updates": updates,
        "train": train_metrics,
        "validation_full_batch": task.batch_size,
        "validation_tail_batch": validation_tail_size,
        "validation": validation,
        "passed": True,
    }


def assert_checkpoint_state_exact(actual: Any, expected: Any, path: str = "state") -> None:
    """Serialization restores exactly; CUDA arithmetic tolerance never applies here."""
    if torch.is_tensor(expected):
        if (
            not torch.is_tensor(actual)
            or actual.dtype != expected.dtype
            or not torch.equal(actual.detach().cpu(), expected.detach().cpu())
        ):
            raise RuntimeError(f"checkpoint restoration differs: {path}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError(f"checkpoint restoration keys differ: {path}")
        for key in expected:
            assert_checkpoint_state_exact(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise RuntimeError(f"checkpoint restoration sequence differs: {path}")
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            assert_checkpoint_state_exact(value, reference, f"{path}/{index}")
    elif actual != expected:
        raise RuntimeError(f"checkpoint restoration differs: {path}")


def continuation_metrics(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    """Bound CUDA replay noise while requiring exact CPU replay and integer state."""
    if actual.keys() != expected.keys():
        raise RuntimeError("checkpoint continuation tensor keys differ")
    difference_squared = reference_squared = maximum_absolute = 0.0
    elementwise_passed = True
    count = 0
    # EVM's repeated CUDA replay differs even after bitwise-exact restoration.
    # The observed max was 3.052e-6; retain a small absolute bound AND a
    # 0.05% aggregate norm bound, calculated separately for each state family.
    atol = 1e-5 if device.type == "cuda" else 0.0
    rtol = 1e-5 if device.type == "cuda" else 0.0
    for key, value in actual.items():
        reference = expected[key]
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise RuntimeError(f"checkpoint continuation dtype/shape differs: {key}")
        actual_cpu, reference = value.detach().cpu(), reference.detach().cpu()
        if not actual_cpu.is_floating_point():
            assert_checkpoint_state_exact(actual_cpu, reference, key)
            continue
        a, b = actual_cpu.double(), reference.double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError(f"nonfinite checkpoint continuation: {key}")
        difference = a - b
        maximum_absolute = max(maximum_absolute, float(difference.abs().max()))
        difference_squared += float(difference.square().sum())
        reference_squared += float(b.square().sum())
        elementwise_passed &= bool((difference.abs() <= atol + rtol * b.abs()).all())
        count += value.numel()
    relative_l2 = math.sqrt(difference_squared / max(reference_squared, 1e-300))
    return {
        "elements": count,
        "maximum_absolute": maximum_absolute,
        "relative_l2": relative_l2,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "maximum_relative_l2": 5e-4 if device.type == "cuda" else 0.0,
        "passed": elementwise_passed and relative_l2 <= (5e-4 if device.type == "cuda" else 0.0),
    }


def smoke(args: argparse.Namespace, task: runtime.BaselineTask) -> dict[str, Any]:
    """Separate real-data diagnostic; never creates a resumable training checkpoint."""
    if args.stop_file.is_file():
        raise runtime.TrainingInterrupted("stop requested before smoke")
    if task.checkpoint_path.exists() or task.result_path.exists():
        raise ValueError("smoke must use a separate diagnostic output root")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(task.seed)
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(task.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    model = make_builder(args, task)(task.model_key, task.source_root, 1000).to(device)
    model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    runtime._compile_model_in_place(model, task.execution)
    optimizer = runtime._build_optimizer(model, task.learning_rate, device)
    bundle = runtime._build_loaders(task, device)
    count = args.smoke_updates * task.gradient_accumulation_steps
    if len(bundle.train) < count:
        raise ValueError("not enough data for requested smoke updates")

    class FirstBatches:
        def __len__(self) -> int:
            return count

        def __iter__(self) -> Iterator[Any]:
            import itertools

            return itertools.islice(bundle.train, count)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    metrics, steps = runtime._train_epoch(
        model,
        cast("DataLoader[Any]", cast("object", FirstBatches())),
        optimizer,
        runtime._make_mixup(task),
        task,
        device,
        0,
        stop_requested=args.stop_file.is_file,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.monotonic() - started
    if steps != args.smoke_updates or not math.isfinite(metrics["loss"]):
        raise RuntimeError("smoke update count/loss is invalid")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("nonfinite parameters after smoke")
    training_memory = memory_metrics()
    # Freeze one update's input tensors: live persistent data workers cannot be
    # rewound mid-epoch. Replay checks model/optimizer/Mixup/dropout RNG while
    # full-run resume remains at complete-epoch boundaries in the shared runtime.
    import itertools

    batches = list(itertools.islice(bundle.train, task.gradient_accumulation_steps))
    path = task.output_dir / "smoke" / "checkpoint.pt"
    contract = runtime._contract(task, device)
    runtime._atomic_torch(
        path,
        {
            "contract_sha256": runtime._sha256_payload(contract),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": runtime._capture_rng(bundle, device),
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["contract_sha256"] != runtime._sha256_payload(contract):
        raise RuntimeError("smoke checkpoint contract mismatch")

    class ReplayBatches:
        def __len__(self) -> int:
            return len(batches)

        def __iter__(self) -> Iterator[Any]:
            return iter(tuple(tuple(t.clone() for t in batch) for batch in batches))

    replay_loader = cast("DataLoader[Any]", cast("object", ReplayBatches()))
    assert_checkpoint_state_exact(model.state_dict(), payload["model"], "saved_model")
    assert_checkpoint_state_exact(optimizer.state_dict(), payload["optimizer"], "saved_optimizer")
    assert_checkpoint_state_exact(runtime._capture_rng(bundle, device), payload["rng"], "saved_rng")
    _, continued_step = runtime._train_epoch(
        model,
        replay_loader,
        optimizer,
        runtime._make_mixup(task),
        task,
        device,
        steps,
        stop_requested=args.stop_file.is_file,
    )
    expected_model = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    runtime._restore_rng(payload["rng"], bundle, device)
    assert_checkpoint_state_exact(model.state_dict(), payload["model"], "restored_model")
    assert_checkpoint_state_exact(
        optimizer.state_dict(), payload["optimizer"], "restored_optimizer"
    )
    assert_checkpoint_state_exact(
        runtime._capture_rng(bundle, device), payload["rng"], "restored_rng"
    )
    _, resumed_step = runtime._train_epoch(
        model,
        replay_loader,
        optimizer,
        runtime._make_mixup(task),
        task,
        device,
        steps,
        stop_requested=args.stop_file.is_file,
    )
    if resumed_step != continued_step or resumed_step != steps + 1:
        raise RuntimeError("checkpoint continuation update count differs")
    actual_model = model.state_dict()
    replay = {"model": continuation_metrics(actual_model, expected_model, device)}
    parameter_names = {name for name, _ in model.named_parameters()}
    replay["parameter_updates"] = continuation_metrics(
        {
            name: actual_model[name].detach().cpu() - payload["model"][name]
            for name in parameter_names
        },
        {name: expected_model[name] - payload["model"][name] for name in parameter_names},
        device,
    )
    actual_optimizer = optimizer.state_dict()
    assert_checkpoint_state_exact(
        actual_optimizer["param_groups"], expected_optimizer["param_groups"], "optimizer_groups"
    )
    if actual_optimizer["state"].keys() != expected_optimizer["state"].keys():
        raise RuntimeError("checkpoint continuation optimizer keys differ")
    families: dict[str, dict[str, torch.Tensor]] = {}
    expected_families: dict[str, dict[str, torch.Tensor]] = {}
    for key, values in actual_optimizer["state"].items():
        if values.keys() != expected_optimizer["state"][key].keys():
            raise RuntimeError("checkpoint continuation optimizer state fields differ")
        for name, value in values.items():
            reference = expected_optimizer["state"][key][name]
            if name == "step" or not torch.is_tensor(value):
                assert_checkpoint_state_exact(value, reference, f"optimizer/{key}/{name}")
            else:
                families.setdefault(name, {})[str(key)] = value
                expected_families.setdefault(name, {})[str(key)] = reference
    for name, values in families.items():
        replay[f"optimizer_{name}"] = continuation_metrics(values, expected_families[name], device)
    runtime._atomic_json(task.output_dir / "smoke" / "continuation-metrics.json", replay)
    if not all(row["passed"] for row in replay.values()):
        raise RuntimeError(
            "checkpoint next-update replay exceeds measured CUDA tolerance; see continuation-metrics.json"
        )
    edges = edge_smoke(
        model,
        optimizer,
        bundle,
        task,
        device,
        batches[0],
        args.stop_file.is_file,
    )
    result = {
        "status": "smoke_passed",
        "updates": steps,
        "train": metrics,
        "seconds_including_compile": duration,
        "images_per_second_including_compile": metrics["examples"] / duration,
        "diagnostic_only": True,
        "checkpoint_restoration_exact": True,
        "checkpoint_next_update_replay_passed": True,
        "continuation_update": resumed_step,
        "continuation_metrics": replay,
        "full_and_remainder_batches": edges,
        "note": "Short-loader smoke schedule; not a training checkpoint or steady-state benchmark.",
        **training_memory,
    }
    runtime._atomic_json(task.output_dir / "smoke" / "result.json", result)
    print("BASELINE_SMOKE=" + json.dumps(result), flush=True)
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=501)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args(argv)
    if args.smoke_updates < 0 or (args.smoke_updates and args.resume):
        parser.error("smoke updates must be nonnegative and cannot be resumed")
    return args


if __name__ == "__main__":
    cli = arguments()
    cli.stop_file.parent.mkdir(parents=True, exist_ok=True)

    def request_stop(_signum: int, _frame: object) -> None:
        cli.stop_file.touch()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    try:
        train(cli)
    except runtime.TrainingInterrupted:
        print("BASELINE_STOPPED_WITHOUT_VALIDATION_OR_BACKUP", flush=True)
        raise SystemExit(130) from None
