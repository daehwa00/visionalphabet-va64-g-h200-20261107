"""Inspect register use for backbone candidates and the unchanged vector stem."""

import argparse
import itertools
import json
import math
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.fused_backbone_ops import _gate_kernel, _rms_backward_kernel
from lnet.fused_stem_ops import _compile_ptx

# ruff: noqa: S603, T201


def assemble(ptx: bytes, arch: int, ptxas: str, folder: Path) -> dict:
    source = folder / "kernel.ptx"
    source.write_bytes(ptx)
    target = re.search(rb"\.target (sm_\w+)", ptx)
    sm = target[1].decode() if target else f"sm_{arch}"
    result = subprocess.run(
        [ptxas, "-v", f"-arch={sm}", str(source), "-o", str(folder / "kernel.cubin")],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", result.stderr)
    if not spills:
        raise RuntimeError(result.stderr)
    return {
        "spill_stores": max(int(store) for store, _ in spills),
        "spill_loads": max(int(load) for _, load in spills),
        "ptxas": result.stderr,
    }


def candidate(arch: int, dtype: str, width: int, name: str) -> bytes:
    if name == "rms":
        kernel = _rms_backward_kernel
        signature = [f"*{dtype}", f"*{dtype}", "*fp32", f"*{dtype}", f"*{dtype}", "*fp32", "i32"]
        block_columns = triton.next_power_of_2(width)
        constants = {
            "width": width, "block_rows": min(16, 4096 // block_columns),
            "block_columns": block_columns, "tiles": 4,
        }
    else:
        kernel = _gate_kernel
        signature = [f"*{dtype}"] * 3 + ["i32"]
        vector = math.gcd(width, 4 if dtype == "fp32" else 8)
        constants = {
            "hidden_groups": width // vector, "block": 1024,
            "vector": vector, "backward": name == "gate-backward",
        }
    signature.extend(["constexpr"] * len(constants))
    tensor_attrs = {
        (i,): [["tt.divisibility", 16]] for i, spec in enumerate(signature) if spec.startswith("*")
    }
    source = ASTSource(
        kernel, dict(zip(kernel.arg_names, signature, strict=True)), constants, attrs=tensor_attrs,
    )
    compiled = triton.compile(
        source, target=GPUTarget("cuda", arch, 32),
        options={"enable_fp_fusion": name != "rms"},
    )
    return compiled.asm["ptx"].encode()


def dump_sass(args: argparse.Namespace, folder: Path, label: str) -> None:
    with (args.sass_out / f"{label}.sass").open("w") as stream:
        subprocess.run(
            [args.cuobjdump, "--dump-sass", str(folder / "kernel.cubin")],
            stdout=stream, check=True, text=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--sass-out", type=Path)
    args = parser.parse_args()
    if args.sass_out:
        args.sass_out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="va-t62-ptxas-") as directory:
        folder = Path(directory)
        for arch, dtype in itertools.product((86, 89, 90), ("bf16", "fp16", "fp32")):
            for width, name in itertools.product(
                (31, 64, 96, 128, 144, 160, 192, 256, 320, 384, 640, 1024),
                ("gate-forward", "gate-backward", "rms"),
            ):
                ptx = candidate(arch, dtype, width, name)
                print(json.dumps({
                    "arch": arch, "dtype": dtype, "width": width, "op": name,
                    **assemble(ptx, arch, args.ptxas, folder),
                }), flush=True)
                if args.sass_out and width in (31, 128, 192, 640):
                    dump_sass(args, folder, f"{arch}-{dtype}-{width}-{name}")
            stem_type = {"bf16": "__nv_bfloat16", "fp16": "__half", "fp32": "float"}[dtype]
            for width in (32, 48, 64, 96, 160):
                stem_ptx = _compile_ptx(
                    divmod(arch, 10), width, (stem_type, "float", "float", stem_type),
                    vector=True, conv_bias_type="float",
                ).value
                print(json.dumps({
                    "arch": arch, "dtype": dtype, "width": width, "op": "stem-baseline",
                    **assemble(stem_ptx, arch, args.ptxas, folder),
                }), flush=True)
                if args.sass_out and width in (32, 64, 160):
                    dump_sass(args, folder, f"{arch}-{dtype}-{width}-stem")


if __name__ == "__main__":
    main()
