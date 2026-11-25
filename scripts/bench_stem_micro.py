"""Benchmark eager Stage-0 Conv-LayerNorm-GELU with cold CUDA L2."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet.cuda_nvrtc import _check_cu, _load_cuda_driver
from lnet.image_layers import STEM_STAGE0_MODES, LayerNorm2d, _fused_stem_block

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
FIELDS = ("out", "dW", "dbias", "dln_w", "dln_b", "dX")


def l2_buffer() -> Tensor:
    size = ctypes.c_int()
    driver = _load_cuda_driver()
    # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
    _check_cu(
        driver.cuDeviceGetAttribute(ctypes.byref(size), 38, torch.cuda.current_device()),
        "cuDeviceGetAttribute(L2_CACHE_SIZE)",
    )
    return torch.empty(max(64 << 20, 2 * size.value), device="cuda", dtype=torch.uint8)


class Workload:
    def __init__(
        self, args: argparse.Namespace, batch: int, dtype: torch.dtype, *, input_grad: bool
    ) -> None:
        torch.manual_seed(4160)
        self.conv = nn.Conv2d(
            args.c_in, args.c_out, 3, args.stride, 1, device="cuda",
        ).to(memory_format=torch.channels_last)
        self.norm = LayerNorm2d(args.c_out).cuda()
        self.x = torch.randn((batch, args.c_in, args.height, args.width), device="cuda")
        self.x = self.x.contiguous(memory_format=torch.channels_last).requires_grad_(input_grad)
        self.params = (*self.conv.parameters(), *self.norm.parameters())
        if input_grad:
            self.params = (*self.params, self.x)
        self.dtype = dtype
        self.dy = torch.randn(
            (batch, args.c_out, (args.height - 1) // args.stride + 1,
             (args.width - 1) // args.stride + 1),
            device="cuda",
        ).contiguous(memory_format=torch.channels_last).to(dtype)

    def forward(self, mode: str) -> Tensor:
        if mode == "fp32":
            return functional.gelu(self.norm(self.conv(self.x)))
        with torch.autocast("cuda", dtype=self.dtype):
            return _fused_stem_block(self.x, self.conv, self.norm, stage0_mode=mode)

    def backward(self, output: Tensor) -> tuple[Tensor, ...]:
        return torch.autograd.grad(output, self.params, self.dy)


def measure(
    workload: Workload, mode: str, phase: str, warmup: int, iters: int, flush: Tensor | None
) -> float:
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for i in range(-warmup, iters):
        if phase == "bwd":
            output = workload.forward(mode)
        if i >= 0:
            if flush is not None:
                flush.zero_()
            start, stop = events[i]
            start.record()
        if phase != "bwd":
            output = workload.forward(mode)
        if phase != "fwd":
            workload.backward(output)
        if i >= 0:
            stop.record()
        del output
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(stop) for start, stop in events)


def errors(actual: tuple[Tensor, ...], expected: tuple[Tensor, ...]) -> dict:
    result = {}
    for field, got, expected_tensor in zip(FIELDS[:len(actual)], actual, expected, strict=True):
        ref = expected_tensor.float()
        delta = (got.float() - ref).abs()
        result[field] = {
            "max_abs": delta.max().item(),
            "max_rel": (delta / ref.abs().clamp_min(1e-6)).max().item(),
            "rel_l2": (delta.norm() / ref.norm().clamp_min(1e-8)).item(),
            "finite": bool(torch.isfinite(got).all()),
        }
    return result


def parity(args: argparse.Namespace) -> list[dict]:
    results = []
    for name, dtype in DTYPES.items():
        for input_grad in (False, True):
            batch = min(args.batch, args.parity_batch)
            workload = Workload(args, batch, dtype, input_grad=input_grad)
            values = {}
            for mode in dict.fromkeys(("off", "fp32", *args.modes)):
                print(f"parity {name} dX={input_grad} B={batch} {mode}", file=sys.stderr, flush=True)
                output = workload.forward(mode)
                values[mode] = (output.detach(), *workload.backward(output))
                del output
            results.append({
                "dtype": name,
                "batch": batch,
                "input_grad": input_grad,
                "modes": {
                    mode: {
                        "off": errors(values[mode], values["off"]),
                        "fp32": errors(values[mode], values["fp32"]),
                    }
                    for mode in args.modes
                },
            })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default=",".join(STEM_STAGE0_MODES))
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--c-in", type=int, default=3)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--c-out", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--parity-batch", type=int, default=2, help="maximum batch for parity")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    args.modes = tuple(dict.fromkeys(part.strip() for part in args.modes.split(",")))
    if set(args.modes) - set(STEM_STAGE0_MODES):
        parser.error(f"--modes must be from {', '.join(STEM_STAGE0_MODES)}")
    if min(args.batch, args.height, args.width, args.c_in, args.c_out, args.stride,
           args.iters, args.parity_batch) <= 0 or args.warmup < 0:
        parser.error("expected positive dimensions and iteration count, and nonnegative warmup")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for timing and device parity")
    flush = l2_buffer() if args.flush_l2 else None
    workload = Workload(args, args.batch, DTYPES[args.dtype], input_grad=False)
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": [args.batch, args.c_in, args.height, args.width, args.c_out, args.stride],
        "dtype": args.dtype,
        "flush_bytes": flush.numel() if flush is not None else 0,
        "warmup": args.warmup,
        "iters": args.iters,
        "median_ms": {},
    }
    for mode in args.modes:
        print(f"timing {mode}", file=sys.stderr, flush=True)
        result["median_ms"][mode] = {
            phase: measure(workload, mode, phase, args.warmup, args.iters, flush)
            for phase in ("fwd", "bwd", "fwd_bwd")
        }
    del workload, flush
    if args.parity:
        torch.backends.cudnn.allow_tf32 = False
        result["parity"] = parity(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
