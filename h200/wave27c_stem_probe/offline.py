"""Compile Wave 27C Tensor Core kernels offline and record resource usage."""

# ruff: noqa: S603, T201
import argparse
import itertools
import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.fused_stem_ops import _stem_schedule, stem_conv_tensor_reduce, stem_conv_tensor_stages


def audit(fn, arch, pointers, constants, warps, stages, output):  # noqa: ANN001, ANN201
    source = ASTSource(
        fn,
        dict(zip(fn.arg_names, [*pointers, *["constexpr"] * len(constants)], strict=True)),
        dict(zip(fn.arg_names[len(pointers) :], constants, strict=True)),
        attrs={(i,): [["tt.divisibility", 16]] for i in range(len(pointers))},
    )
    image = triton.compile(
        source,
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": warps, "num_stages": stages},
    )
    output.write_bytes(image.asm["cubin"])
    usage = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "-res-usage", output],
        text=True,
    )
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", usage)
    if match is None:
        raise ValueError(usage)
    registers, stack, static_shared, local = map(int, match.groups())
    shared = static_shared + image.metadata.shared
    warp_limit = {86: 48, 89: 48, 90: 64}[arch]
    block_limit = {86: 16, 89: 24, 90: 32}[arch]
    shared_capacity = 228 if arch == 90 else 100
    resident = min(
        block_limit,
        warp_limit // warps,
        65536 // (warps * triton.cdiv(registers, 8) * 256),
        shared_capacity * 1024 // (shared + 1024),
    )
    return {"regs": registers, "stack": stack, "local": local, "smem": shared, "cta_sm": resident}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--arch", type=int, choices=(86, 89, 90), action="append")
    args = parser.parse_args()
    spilled = False
    with TemporaryDirectory(prefix="va-wave27c-") as root:
        output = Path(root) / "stem.cubin"
        for arch, mode, dtype in itertools.product(
            args.arch or (86, 89, 90),
            ("off", "stage0_opt", "stage0_v2", "hybrid"),
            ("bf16", "fp16"),
        ):
            hybrid = mode == "hybrid"
            ci, co = (32, 64) if hybrid else (3, 32)
            standard = (112, 112) if hybrid else (224, 224)
            combos = itertools.product(
                (True,) if args.quick else (False, True),
                (True,) if args.quick else (False, True),
                (False, True),
                (standard,) if args.quick else (standard, (113, 127)),
                (1024,) if args.quick else (128, 1024),
            )
            for layout, bias, backward, (h, w), batch in combos:
                pixels = batch * ((h + 1) // 2) * ((w + 1) // 2)
                block, warps, stages, tile_reduce, _ = _stem_schedule(
                    mode, backward=backward, channels_last=layout
                )
                ptr = [f"*{dtype}"] * 2 + ["*fp32"] * 3 + [f"*{dtype}"] * 2 + ["*fp32"] * 2
                res = audit(
                    stem_conv_tensor_stages,
                    arch,
                    ptr,
                    (
                        h,
                        w,
                        ci,
                        co,
                        pixels,
                        1e-5,
                        layout,
                        bias,
                        backward,
                        block,
                        32,
                        tile_reduce,
                    ),
                    warps,
                    stages,
                    output,
                )
                print(
                    json.dumps(
                        dict(
                            arch=arch,
                            mode=mode,
                            dtype=dtype,
                            cl=layout,
                            bias=bias,
                            backward=backward,
                            h=h,
                            w=w,
                            batch=batch,
                            **res,
                        )
                    ),
                    flush=True,
                )
                spilled |= bool(res["stack"] or res["local"])
            sms = {86: 80, 89: 128, 90: 132}[arch]
            waves = _stem_schedule(mode, backward=True)[-1]
            workers = waves * sms if waves else 1024
            for bias in (False, True):
                res = audit(
                    stem_conv_tensor_reduce,
                    arch,
                    ["*fp32", f"*{dtype}", *(["*fp32"] * 3)],
                    (0 if hybrid else ci, co, bias, workers, triton.next_power_of_2(workers)),
                    8 if workers > 1024 else 4,
                    3,
                    output,
                )
                print(
                    json.dumps(
                        dict(
                            arch=arch,
                            mode=mode,
                            dtype=dtype,
                            bias=bias,
                            reduce=True,
                            workers=workers,
                            **res,
                        )
                    ),
                    flush=True,
                )
                spilled |= bool(res["stack"] or res["local"])
    if spilled:
        raise SystemExit("Stack or local memory detected")


if __name__ == "__main__":
    main()
