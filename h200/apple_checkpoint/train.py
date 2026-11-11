#!/usr/bin/env python3
"""Scratch Pythia-70M + AV + gamma*(V D)U on sequential preshuffled Pile.

This is a native PyTorch runtime, not a DeepSpeed bitwise reproduction. FP32
parameters/moments, FP16 autocast and dynamic scaling preserve the declared
recipe; scaler overflow hysteresis and kernel reduction orders can differ.
No pretrained weights, synthetic training, or canonical V-AV fused kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SEQUENCE = 2048
GLOBAL_BATCH = 1024
DEFAULT_STEPS = 47_684
SCHEDULE_STEPS = 143_000
WARMUP_STEPS = 1_430
BASE_LR = 1e-3
BRANCH_LR = 1e-4
MIN_LR = 1e-4
VOCAB_SIZE = 50_304


def model_config():
    from transformers import GPTNeoXConfig
    config = GPTNeoXConfig(
        vocab_size=VOCAB_SIZE, hidden_size=512, num_hidden_layers=6,
        num_attention_heads=8, intermediate_size=2048,
        hidden_act="gelu", rotary_pct=0.25, rotary_emb_base=10000,
        max_position_embeddings=2048, layer_norm_eps=1e-5,
        use_parallel_residual=True, tie_word_embeddings=False,
        attention_dropout=0.0, hidden_dropout=0.0, use_cache=False,
        bos_token_id=0, eos_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return config


def paper_initialize(model):
    """NeoX small_init everywhere; wang_init on residual output projections."""
    hidden = model.config.hidden_size
    small_std = math.sqrt(2.0 / (5.0 * hidden))
    wang_std = 2.0 / model.config.num_hidden_layers / math.sqrt(hidden)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                is_residual_output = name.endswith(("attention.dense", "mlp.dense_4h_to_h"))
                module.weight.normal_(0.0, wang_std if is_residual_output else small_std)
                if getattr(module, "bias", None) is not None:
                    module.bias.zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()
    return {"small_std": small_std, "wang_std": wang_std}


def build_model(seed=1234, config=None, branch_source="value"):
    from transformers import GPTNeoXForCausalLM
    from lrcp.pythia import attach_lrcp, set_lrcp_stats_enabled
    from lrcp.source_controls import replace_lrcp_source
    torch.manual_seed(seed)
    model = GPTNeoXForCausalLM(config if config is not None else model_config())
    paper_initialize(model)
    attach_lrcp(model, layers="all", rank=4, gamma=1.0, down_init_std=0.02, up_init_std=0.0)
    if branch_source not in {"value", "residual"}:
        raise ValueError("Unknown branch source")
    count = replace_lrcp_source(model, "value") if branch_source == "value" else model.config.num_hidden_layers
    if count != model.config.num_hidden_layers:
        raise RuntimeError(f"Expected one V-only branch per layer, found {count}")
    set_lrcp_stats_enabled(model, False)
    model.train()
    return model


def optimizer_groups(model):
    """Match NeoX: LayerNorm/bias no WD; branch LR 1e-4 and no WD."""
    no_decay_ids = {
        id(p) for module in model.modules() if isinstance(module, torch.nn.LayerNorm)
        for p in module.parameters(recurse=False)
    }
    groups = {"base_decay": [], "base_no_decay": [], "branch": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lrcp." in name:
            key = "branch"
        elif name.endswith(".bias") or id(parameter) in no_decay_ids:
            key = "base_no_decay"
        else:
            key = "base_decay"
        groups[key].append(parameter)
    return [
        {"params": params, "name": name, "lr": BRANCH_LR if name == "branch" else BASE_LR,
         "lr_scale": 0.1 if name == "branch" else 1.0,
         "weight_decay": 0.1 if name == "base_decay" else 0.0}
        for name, params in groups.items() if params
    ]


def scheduled_lr(step):
    """Exact NeoX AnnealingLR cosine/floor formula; first optimizer LR is zero.

    NeoX get_learning_rate_scheduler initializes last_iter=0. DeepSpeed advances
    the scheduler after a successful optimizer update, so overflow skips do not
    advance this index. The original end_iter denominator/clamp are retained.
    """
    step = max(0, int(step))
    effective = min(step, SCHEDULE_STEPS - WARMUP_STEPS)
    if step <= WARMUP_STEPS:
        return BASE_LR * effective / WARMUP_STEPS
    effective -= WARMUP_STEPS
    return max(BASE_LR / 2.0 * (math.cos(math.pi * effective / SCHEDULE_STEPS) + 1.0), MIN_LR)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class StopRequested(Exception):
    pass


class Stopper:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.signal_number = None

    def install(self):
        def stop(signum, _frame):
            self.signal_number = signum
            # Raise immediately even in a network read; do not checkpoint/evaluate.
            raise StopRequested(f"signal {signum}")
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

    def check(self):
        if self.signal_number is not None or (self.path is not None and self.path.exists()):
            raise StopRequested("stop requested")


def source_hash():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root / "stream.py", root / "checkpointing.py", *sorted((root / "lrcp").glob("*.py"))]
    for path in paths:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def contract(args, model):
    return {
        "model": "Pythia-70M scratch", "branch_source": args.branch_source,
        "equation": "AV + gamma * ((V @ D) @ U)" if args.branch_source == "value" else "AV + gamma * (((V-AV) @ D) @ U)", "rank": 4, "gamma": 1.0,
        "seed": args.seed, "initialization": {"base": "small_init", "residual_output": "wang_init", "D_std": 0.02, "U": 0.0},
        "source_sha256": source_hash(), "microbatch": args.microbatch,
        "data_manifest": args.data_manifest,
        "data_manifest_sha256": hashlib.sha256(Path(args.data_manifest).read_bytes()).hexdigest() if args.data_manifest else None,
        "validation_manifest": args.validation_manifest,
        "validation_manifest_sha256": hashlib.sha256(Path(args.validation_manifest).read_bytes()).hexdigest() if args.validation_manifest else None,
        "gradient_accumulation_steps": GLOBAL_BATCH // args.microbatch,
        "global_batch": GLOBAL_BATCH, "sequence_length": SEQUENCE,
        "steps": args.steps, "tokens_target": args.steps * GLOBAL_BATCH * SEQUENCE,
        "data_order": "official preshuffled sequential; changed from paper random sample access",
        "optimizer": {"name": "fused AdamW", "base_lr": BASE_LR, "branch_lr": BRANCH_LR,
                      "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1,
                      "branch_weight_decay": 0.0, "grad_clip": 1.0},
        "schedule": {"horizon": SCHEDULE_STEPS, "warmup": WARMUP_STEPS, "min_base_lr": MIN_LR,
                     "formula": "NeoX AnnealingLR cosine with min floor; original end_iter denominator/clamp",
                     "index": "completed successful optimizer updates; first update LR=0"},
        "precision": "FP32 parameters/gradients/moments; FP16 autocast; GradScaler4096/growth1000",
        "runtime": "native PyTorch; not DeepSpeed exact; no hysteresis2; no canonical fused branch; no compile",
        "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_policy": "latest+previous resume checkpoints; best+final weights", "synthetic": bool(args.benchmark or args.smoke),
    }



def checkpoint_contract(config, args):
    # Absolute mount paths may differ after recovery; content and recipe may not.
    result = {k: v for k, v in config.items() if k not in {"data_manifest", "validation_manifest", "validation_manifest_sha256"}}
    if args.validation_manifest:
        meta = json.loads(Path(args.validation_manifest).read_text())
        result["validation_token_sha256"] = meta["token_file_sha256"]
    return result


def loss_for_batch(model, tokens):
    # Every row contains S+1 tokens; all S positions have a next-token target.
    logits = model(input_ids=tokens[:, :-1].contiguous(), use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))


def evaluate(model, manifest_path, microbatch, device, stopper):
    """Evaluate a disjoint official validation prefix without resetting train RNG."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("dtype") != "uint16_le":
        raise ValueError("Validation manifest dtype must be uint16_le")
    token_file = Path(manifest["token_file"])
    values = np.memmap(token_file, dtype="<u2", mode="r")
    needed = GLOBAL_BATCH * SEQUENCE + 1
    if len(values) < needed or int(manifest["tokens"]) < needed:
        raise ValueError(f"Validation requires at least {needed} tokens")
    summed = torch.zeros((), dtype=torch.float64, device=device)
    began = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in range(0, GLOBAL_BATCH, microbatch):
                stopper.check()
                # Adjacent sequences overlap one boundary token, exactly as train.
                offsets = (np.arange(row, row + microbatch)[:, None] * SEQUENCE
                           + np.arange(SEQUENCE + 1)[None, :])
                tokens = torch.from_numpy(np.asarray(values[offsets], dtype=np.int64)).to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_for_batch(model, tokens)
                torch._assert_async(torch.isfinite(loss), "Nonfinite validation loss")
                summed.add_(loss.double() * microbatch)
            mean = float((summed / GLOBAL_BATCH).cpu())
        return {"val_loss": mean, "val_perplexity": math.exp(mean),
                "validation_tokens": GLOBAL_BATCH * SEQUENCE,
                "validation_seconds": time.perf_counter() - began}
    finally:
        model.train(was_training)


def run(args):
    if args.microbatch <= 0 or GLOBAL_BATCH % args.microbatch:
        raise ValueError("microbatch must divide global batch 1024")
    if args.stop_after is not None and (args.stop_after < 1 or args.stop_after > args.steps):
        raise ValueError("--stop-after must be within the requested total budget")
    if args.steps <= 0 or args.steps > DEFAULT_STEPS:
        raise ValueError(f"steps must be in [1, {DEFAULT_STEPS}]")
    if not (args.benchmark or args.smoke) and not args.data_manifest:
        raise ValueError("Real training requires --data-manifest; synthetic data is benchmark/smoke only")
    if not (args.benchmark or args.smoke) and not args.validation_manifest:
        raise ValueError("Real training requires --validation-manifest")
    if args.benchmark and (args.warmup < 1 or args.measure < 1):
        raise ValueError("benchmark warmup and measure must be positive")
    stopper = Stopper(args.stop_file)
    stopper.install()
    stopper.check()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    # Existing run logs must never be silently overwritten or treated as resume.
    if any((output / name).exists() for name in ("resolved-config.json", "result.json", "steps.jsonl")):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for FP16 training; use --validate for CPU checks")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(args.seed, branch_source=args.branch_source).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model), betas=(0.9, 0.95), eps=1e-8, fused=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.0, growth_interval=1000)
    config = contract(args, model)
    atomic_json(output / "resolved-config.json", config)
    stream = None
    if not (args.benchmark or args.smoke):
        from stream import TokenStream
        stream = TokenStream(args.data_manifest, args.cache_root, args.cache_bytes, args.stop_file)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    synthetic = None
    if args.benchmark or args.smoke:
        synthetic = torch.randint(VOCAB_SIZE, (args.microbatch, SEQUENCE + 1), generator=generator, device=device)
    from checkpointing import save_training, save_weights, restore_training
    saved_contract = checkpoint_contract(config, args)
    gas = GLOBAL_BATCH // args.microbatch
    skipped = 0
    consecutive_skips = 0
    last_validation = None
    steps_done = 0
    micro_done = 0
    consumed = 0
    result = {"status": "running", "mode": "benchmark" if args.benchmark else "smoke" if args.smoke else "train"}
    best_val = float("inf")
    previous_elapsed = 0.0
    if args.resume:
        if args.benchmark or args.smoke:raise ValueError("Resume is only for real training")
        saved = restore_training(args.resume, model, optimizer, scaler, contract=saved_contract, best_output=output / "checkpoints/best.pt")
        steps_done, consumed, skipped = saved["global_step"], saved["tokens_seen"], saved["skipped_optimizer_steps"]
        micro_done = steps_done * gas
        consecutive_skips = saved.get("consecutive_skips", 0)
        best_val = saved.get("best_val") if saved.get("best_val") is not None else float("inf")
        previous_elapsed = saved.get("elapsed_seconds", 0.0)
        if steps_done >= args.steps:raise ValueError("Checkpoint already reached the requested budget")
    start = time.perf_counter()
    last_log = start
    last_log_tokens = consumed
    window_loss = torch.zeros((), dtype=torch.float64, device=device)
    window_micros = 0
    step_loss = torch.zeros((), dtype=torch.float64, device=device)
    benchmark_start = None
    try:
        optimizer.zero_grad(set_to_none=True)
        if args.benchmark:
            # Time complete accumulation groups so optimizer cost is represented.
            args.warmup = math.ceil(args.warmup / gas) * gas
            args.measure = math.ceil(max(args.measure, gas) / gas) * gas
        if stream is not None:
            last_validation = evaluate(model, args.validation_manifest, args.microbatch, device, stopper)
            last_validation["validation_global_step"] = steps_done
            with (output / "validation.jsonl").open("a", buffering=1) as validations:
                validations.write(json.dumps({"global_step": steps_done, "tokens_seen": consumed, **last_validation}) + "\n")
            atomic_json(output / "step-progress.json", {"global_step": steps_done, "tokens_seen": consumed, **last_validation})
            print(json.dumps({"global_step": steps_done, "tokens_seen": consumed, **last_validation}), flush=True)
        end_step = args.stop_after if args.stop_after is not None else args.steps
        if end_step <= steps_done:raise ValueError("Stop boundary must be after resumed step")
        total_micros = args.warmup + args.measure if args.benchmark else end_step * gas
        with (output / "steps.jsonl").open("a", buffering=1) as logs:
            for micro in range(micro_done, total_micros):
                stopper.check()
                if args.benchmark and micro == args.warmup:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    benchmark_start = time.perf_counter()
                if micro % gas == 0:
                    step_loss.zero_()
                    lr = scheduled_lr(steps_done - skipped)
                    for group in optimizer.param_groups:
                        group["lr"] = lr * group["lr_scale"]
                if synthetic is None:
                    batch = stream.batch(consumed, args.microbatch, SEQUENCE)
                    if not isinstance(batch, np.ndarray) or batch.shape != (args.microbatch, SEQUENCE + 1):
                        raise ValueError("TokenStream must return [microbatch, seq+1] numpy tokens")
                    tokens = torch.from_numpy(batch).to(device, dtype=torch.long, non_blocking=True)
                else:
                    tokens = synthetic
                stopper.check()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_for_batch(model, tokens)
                torch._assert_async(torch.isfinite(loss), "Nonfinite training loss")
                scaler.scale(loss / gas).backward()
                step_loss.add_(loss.detach().double())
                window_loss.add_(loss.detach().double())
                window_micros += 1
                micro_done += 1
                consumed += args.microbatch * SEQUENCE
                stopper.check()
                if (micro + 1) % gas == 0:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_skip = scaler.get_scale() < previous_scale
                    skipped += int(did_skip)
                    consecutive_skips = consecutive_skips + 1 if did_skip else 0
                    if consecutive_skips >= 10:
                        raise RuntimeError("10 consecutive FP16 optimizer overflow skips; aborting invalid training")
                    optimizer.zero_grad(set_to_none=True)
                    steps_done += 1
                    if steps_done % args.log_every == 0 or (not args.benchmark and steps_done == end_step):
                        values = torch.stack((step_loss / gas, window_loss / window_micros, norm.double())).cpu().tolist()
                        now = time.perf_counter()
                        record = {"global_step": steps_done, "step": steps_done, "loss": values[0],
                                  "loss_avg": values[1], "grad_norm": values[2] if math.isfinite(values[2]) else None, "lr": lr,
                                  "branch_lr": lr * 0.1, "tokens_seen": consumed,
                                  "optimizer_update_skipped": did_skip,
                                  "tokens_per_second": (consumed-last_log_tokens)/(now-last_log),
                                  "elapsed_seconds": previous_elapsed+now-start, "skipped_optimizer_steps": skipped,
                                  "loss_scale": scaler.get_scale(), "microbatch": args.microbatch,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **(last_validation or {})}
                        if stream is not None:
                            record.update({f"data_{key}": value for key, value in dict(stream.stats).items()})
                        logs.write(json.dumps(record, allow_nan=False) + "\n")
                        atomic_json(output / "step-progress.json", record)
                        print(json.dumps(record, allow_nan=False), flush=True)
                        window_loss.zero_()
                        window_micros = 0
                        last_log, last_log_tokens = now, consumed
                    if stream is not None and (steps_done % args.validation_every == 0 or steps_done == end_step):
                        stopper.check()
                        last_validation = evaluate(model, args.validation_manifest, args.microbatch, device, stopper)
                        last_validation["validation_global_step"] = steps_done
                        validation_record = {"global_step": steps_done, "tokens_seen": consumed, **last_validation}
                        with (output / "validation.jsonl").open("a", buffering=1) as validations:
                            validations.write(json.dumps(validation_record) + "\n")
                        progress_path = output / "step-progress.json"
                        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                        progress.update(validation_record)
                        atomic_json(progress_path, progress)
                        logs.write(json.dumps(progress, allow_nan=False) + "\n")
                        print(json.dumps(validation_record), flush=True)
                        if last_validation["val_loss"] < best_val:
                            best_val = last_validation["val_loss"]
                            stopper.check()
                            save_weights(output / "checkpoints/best.pt", model, contract=saved_contract, model_config=model.config.to_dict(), progress={"global_step": steps_done, "tokens_seen": consumed, "val_loss": best_val})
                    if stream is not None and (steps_done == 1 or steps_done % args.checkpoint_every == 0 or steps_done == end_step):
                        stopper.check()
                        receipt = save_training(output / "checkpoints/latest.pt", model, optimizer, scaler, contract=saved_contract, model_config=model.config.to_dict(), progress={"global_step": steps_done, "tokens_seen": consumed, "skipped_optimizer_steps": skipped, "consecutive_skips": consecutive_skips, "best_val": best_val if math.isfinite(best_val) else None, "elapsed_seconds": previous_elapsed+time.perf_counter()-start}, best_checkpoint=output / "checkpoints/best.pt")
                        atomic_json(output / "checkpoint-manifest.json", receipt)
            torch.cuda.synchronize()
            end = time.perf_counter()
        branch_up_l2 = float(torch.stack([parameter.detach().float().square().sum() for name, parameter in model.named_parameters() if name.endswith("lrcp.up")]).sum().sqrt().cpu())
        result.update(status="completed" if args.benchmark or steps_done == args.steps else "paused", global_step=steps_done, tokens_seen=consumed,
                      elapsed_seconds=previous_elapsed+end-start, skipped_optimizer_steps=skipped,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      optimizer_updates=steps_done-skipped, final_validation=last_validation, branch_up_l2=branch_up_l2,
                      data_stream_stats=dict(stream.stats) if stream is not None else None)
        if args.benchmark:
            measured = end - benchmark_start
            result.update(warmup_microsteps=args.warmup, measured_microsteps=args.measure,
                          seconds=measured, tokens_per_second=args.measure*args.microbatch*SEQUENCE/measured,
                          mean_microstep_ms=1000*measured/args.measure,
                          projected_100b_hours=100_000_000_000/(args.measure*args.microbatch*SEQUENCE/measured)/3600,
                          timing_note="synthetic GPU fwd/bwd plus optimizer at actual GAS; no data/network; measured complete accumulation groups; warmup/measure rounded up to GAS")
        if stream is not None and steps_done == args.steps:
            stopper.check()
            result["final_weights"] = save_weights(output / "checkpoints/final.pt", model, contract=saved_contract, model_config=model.config.to_dict(), progress={"global_step": steps_done, "tokens_seen": consumed, "final_validation": last_validation})
        atomic_json(output / "result.json", result)
        return 0
    except (StopRequested, KeyboardInterrupt) as error:
        # No CUDA sync, checkpoint, evaluation or network upload on a stop.
        result.update(status="stopped", reason=str(error), global_step=steps_done,
                      microsteps=micro_done, tokens_seen=consumed, skipped_optimizer_steps=skipped)
        atomic_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
        return 130
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error_type=type(error).__name__, error=str(error),
                      global_step=steps_done, microsteps=micro_done, tokens_seen=consumed)
        atomic_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
        return 42
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      global_step=steps_done, microsteps=micro_done, tokens_seen=consumed)
        atomic_json(output / "result.json", result)
        raise
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()


def validate():
    """Small CPU checks; no model downloads or CUDA use."""
    from transformers import GPTNeoXConfig
    config = GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=128,
                          max_position_embeddings=32, rotary_pct=0.25,
                          tie_word_embeddings=False, attention_dropout=0.0, hidden_dropout=0.0,
                          use_cache=False)
    config._attn_implementation = "sdpa"
    model = build_model(1234, config)
    groups = optimizer_groups(model)
    parameters = [p for group in groups for p in group["params"]]
    assert len(parameters) == len(set(map(id, parameters)))
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    for block in model.gpt_neox.layers:
        branch = block.attention.lrcp
        assert branch.source == "value"
        assert not torch.count_nonzero(branch.up)
        assert not getattr(branch, "use_fused_training", False)
    print(json.dumps({"status": "validated", "parameters": sum(p.numel() for p in model.parameters()),
                      "cpu_only": True, "source": "value", "no_pretrained_weights": True}))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--branch-source", choices=["value", "residual"], default="value")
    result.add_argument("--checkpoint-every", type=int, default=500)
    result.add_argument("--stop-after", type=int, help="Graceful checkpoint boundary for recovery validation; unlike STOP, saves state")
    result.add_argument("--resume")
    result.add_argument("--data-manifest")
    result.add_argument("--validation-manifest")
    result.add_argument("--validation-every", type=int, default=477)
    result.add_argument("--cache-root", default="/app/scratch/apple70m-stream")
    result.add_argument("--cache-bytes", type=int, default=2*1024**3)
    result.add_argument("--output-root", default="/app/output/apple70m-v-only")
    result.add_argument("--microbatch", type=int, default=8)
    result.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--stop-file")
    result.add_argument("--log-every", type=int, default=1)
    result.add_argument("--benchmark", action="store_true")
    result.add_argument("--smoke", action="store_true", help="Synthetic diagnostic only; never reported as training")
    result.add_argument("--warmup", type=int, default=20)
    result.add_argument("--measure", type=int, default=100)
    result.add_argument("--validate", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.benchmark and args.smoke:
        raise SystemExit("Choose --benchmark or --smoke")
    if args.log_every < 1 or args.validation_every < 1 or args.checkpoint_every < 1:
        raise SystemExit("--log-every and --validation-every must be positive")
    try:
        sys.exit(validate() if args.validate else run(args))
    except (StopRequested, KeyboardInterrupt):
        # Also cover stop during model construction, before the loop handler exists.
        sys.exit(130)
    except torch.cuda.OutOfMemoryError as error:
        # OOM can occur during model construction, before the loop handler exists.
        output = Path(args.output_root)
        output.mkdir(parents=True, exist_ok=True)
        report = {"status": "oom", "error_type": type(error).__name__, "error": str(error),
                  "global_step": 0, "tokens_seen": 0}
        atomic_json(output / "result.json", report)
        print(json.dumps(report), flush=True)
        sys.exit(42)
