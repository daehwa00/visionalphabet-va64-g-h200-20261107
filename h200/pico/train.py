"""Train the seed-501 ImageNet-1K pico-v11 model."""

from argparse import Namespace
from functools import partial
from hashlib import sha256
from pathlib import Path

import alphabet_model_builder as builder
import torch
import va_ablation_augment as augment
import va_imagenet100_runtime as runtime

from alphabet.gpu_policy import apply_gpu_policy

VARIANT = "pico-600-opt"
INIT = Path(__file__).with_name("pico_v11_seed501_init.pt")
INIT_SHA = "c49edc4755ddeffaca127863dd709f890212395a5ec49f1036bcee4d12fb13dd"


def build(
    variant: str = VARIANT, config: builder.ComplexScanConfig | None = None
) -> builder.VisionAlphabetIN100:
    if variant != VARIANT or sha256(INIT.read_bytes()).hexdigest() != INIT_SHA:
        raise ValueError("Unexpected variant or seed archive")
    model = builder.build_dw_terminal_rmsnorm(
        config or builder.model_config(output_dim=1000),
        channels=(64, 72, 80, 128),
        poles=(32, 64, 64, 64),
        depth=(2, 2, 6, 2),
        post_hidden_ratio=1.75,
        factorized_reader=True,
        non_redundant_four_directions=True,
    )
    for parent in list(model.modules()):
        if isinstance(parent, builder.ComplexScanStage):
            parent.enable_pole_unclamp_()
    state = torch.load(INIT, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model


def contract(args: Namespace) -> dict:
    result = runtime.base_contract(args, variants=(VARIANT,), seeds=(501,), num_classes=1000)
    result["schema"] = "alphabet.pico-v11.in1k.v1"
    result["recipe"].update(
        learning_rate=0.0025,
        min_lr=1e-6,
        warmup_epochs=20,
        weight_decay=0.035,
        modal_learning_rate_multiplier=1 / 3,
        pole_geometry_learning_rate_multiplier=1.0,
        drop_path_rate=0.0,
        ema_decay=0.99985,
        ema_warmup=True,
        evaluate_first_epoch=True,
        label_smoothing=0.05,
        mixup_alpha=0.2,
        mix_prob=0.5,
        cutmix=False,
        randaugment_num_ops=2,
        randaugment_magnitude=5,
        color_jitter=0.0,
        random_erasing_prob=0.0,
        gpu_randaugment=args.gpu_randaugment,
        tensor_crop=args.tensor_crop,
        compact_manifest=args.compact_manifest,
        clean_loader_workers=args.clean_loader_workers,
        compile_mode=args.compile_mode if args.compile else None,
        clean_rms_precision="float32",
        schedule="optimizer-step linear warmup then cosine",
    )
    result["runtime"].pop("hostname")
    result["recipe"].pop("cpu_affinity")
    for key in ("loader_workers", "validation_loader_workers"):
        result["recipe"].pop(key)
    return result


if __name__ == "__main__":
    apply_gpu_policy(allow_fp8=False, overwrite=True)
    torch.set_float32_matmul_precision("high")
    runtime.main(
        runtime.RunnerBindings(
            variants=(VARIANT,),
            seeds=(501,),
            model_config=builder.ComplexScanConfig,
            build_model=build,
            contract=contract,
            build_optimizer=builder.build_optimizer,
            prepare_model=runtime.prepare_model,
            train_epoch=partial(augment.train_epoch, mix_prob=0.5),
            evaluate=runtime.evaluate,
            wandb_model_metrics=runtime.wandb_model_metrics,
            summarize=runtime.summarize,
        )
    )
