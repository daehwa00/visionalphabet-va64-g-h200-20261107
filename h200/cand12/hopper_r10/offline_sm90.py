"""Compile the actual 1024×1 specializations with ptxas and inspect cubins."""

import json
from pathlib import Path
import re
import subprocess
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.direct_path_scan_kernels import _path_forward_kernel
from lnet.effective_structural_scan import _BACKWARD_VARIANTS
from lnet.r8_horizontal import r8_horizontal_forward
from lnet.r10_horizontal import r10_hfwd56_seg4
from lnet.r10_rms import VALUE_STRIDE, r10_rms56_partial, rms56_partial_rows
from vertical_seg_screen import segmented_kernel

TARGET = GPUTarget("cuda", 90, 32)
CUOBJDUMP = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
B = 1024


def compile_kernel(fn, constants, bf16, scalar, warps, stages):
    signature = {
        name: ("constexpr" if name in constants else "i32" if name in scalar
               else "*bf16" if name in bf16 else "*fp32")
        for name in fn.arg_names
    }
    attrs = {(i,): [["tt.divisibility", 16]] for i, val in enumerate(signature.values()) if val.startswith("*")}
    code = triton.compile(ASTSource(fn, signature=signature, constexprs=constants, attrs=attrs),
                          target=TARGET, options=dict(num_warps=warps, num_stages=stages))
    binary = next(v for k, v in code.metadata_group.items() if k.endswith(".cubin"))
    dump = subprocess.check_output([str(CUOBJDUMP), "--dump-resource-usage", binary], text=True)
    resource = {k.lower(): int(v) for k, v in re.findall(r"(REG|STACK|LOCAL):(\d+)", dump)}
    return dict(resource, shared_bytes=code.metadata.shared, target="cuda:90",
                local_loads=code.asm["ptx"].count("ld.local"),
                local_stores=code.asm["ptx"].count("st.local"))


def test_specs():
    batch_rows = rms56_partial_rows(B)
    assert batch_rows == 6272
    for ds in ((200704, 3584, 64, 1), VALUE_STRIDE):
        yield f"rms56-dy-{ds[1]}", r10_rms56_partial, dict(SD=ds, SI=(3136, 56, 1, 1), H=56, C=64, ROWS=512, CHANNEL_TILE=16), {
            "DY", "X"}, {"N"}, 4, 1, (4, batch_rows)
    for fn, warps, stages, grid in (
        (r10_hfwd56_seg4, 4, 3, (57344, 5)), (r8_horizontal_forward, 4, 1, (5, 57344)),
    ):
        constants = dict(SOURCE_REAL_STRIDE=320, SOURCE_IMAG_STRIDE=320,
                         BLOCK_WIDTH=64, BLOCK_MODES=32, COMBINED_STATES=True)
        if fn is r8_horizontal_forward:
            constants["BLOCK_LINES"] = 1
        yield fn.__name__, fn, constants, {
            "source_real", "source_imag", "positive_real", "positive_imag", "negative_real", "negative_imag",
        }, {"height", "width", "line_count", "modes"}, warps, stages, grid
    for rows in (2, 4):
        for tile in (4, 8):
            fn = segmented_kernel(_path_forward_kernel, True, rows)
            yield f"vfwd-seg{rows}-bm{tile}", fn, dict(H=56, W=56, M=160, BM=tile, EPS=1e-8,
                                                     GROUP=4, DOUBLING=0), {
                "SA", "IA", "SB", "IB", "OR", "OI",
            }, set(), 4, 2 if tile == 8 else 1, (160 // tile * 4, 14336)
            for coarse in (False, True):
                group = 1 if coarse and tile == 4 else 4
                fn = segmented_kernel(_BACKWARD_VARIANTS[3 if coarse else 2], False, rows)
                yield f"vback-c{int(coarse)}-seg{rows}-bm{tile}", fn, dict(
                    height=56, BM=tile, GROUP=group, COARSE=coarse, DOUBLING=1, epsilon=1e-8,
                ), {
                    "source_real_a", "source_imag_a", "source_real_b", "source_imag_b",
                    "grad_collapsed_real", "grad_collapsed_imag", "grad_source_real_a",
                }, {"width", "line_count", "modes", "packed_path_hidden"}, 2 if tile == 4 else 4, (
                    1 if tile == 4 else 2), (160 // tile * group, B * (28 if coarse else 56) // group)


def main(output):
    results = {}
    for name, fn, constants, bf16, scalar, warps, stages, grid in test_specs():
        assert 0 < grid[1] <= 65535
        try:
            proof = compile_kernel(fn, constants, bf16, scalar, warps, stages)
        except Exception as exc:
            proof = {"error": f"{type(exc).__name__}: {exc}"}
        results[name] = dict(proof, grid=grid, num_warps=warps, num_stages=stages)
        print(name, proof, flush=True)
        output.write_text(json.dumps(results, indent=2) + "\n")
    return any("error" in proof for proof in results.values())


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
