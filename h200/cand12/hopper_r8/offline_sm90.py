"""Compile the live screen's candidate matrix for SM90 without a GPU."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet import direct_path_scan_kernels as path
from lnet import effective_structural_scan as structural
from lnet.triton_bidirectional_product_scan import _bidirectional_forward_kernel
from lnet.triton_product_scan_kernels import (
    _product_scan_coarse4_associative_forward_kernel,
    _product_scan_coarse4_associative_backward_kernel,
)
from run_all import ROOT, save
from screen import Config, candidate_kernel, newest, screen_configs

TARGET = GPUTarget("cuda", 90, 32)
CUOBJDUMP = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
SCALARS = {"height", "width", "line_count", "modes", "packed_path_hidden"}
BF16 = {
    "SA", "IA", "SB", "IB", "OR", "OI", "SR", "SI", "STATES",
    "POS_R", "POS_I", "NEG_R", "NEG_I", "DS",
    "source_real", "source_imag", "positive_real", "positive_imag", "negative_real", "negative_imag",
    "collapsed_real", "collapsed_imag", "coarse_real", "coarse_imag",
    "grad_collapsed_real", "grad_collapsed_imag",
}


def case_specs(batch):
    for coarse in (0, 1):
        yield ("H1", 56, coarse), structural._BACKWARD_VARIANTS[3 if coarse else 2], dict(
            height=56, width=56, modes=160, line_count=batch * 56, epsilon=1e-8, packed_path_hidden=8,
            BM=4, GROUP=1 if coarse else 4, COARSE=bool(coarse), DOUBLING=1,
        )
    yield ("H3", 56, 0), path._path_forward_kernel, dict(
        H=56, W=56, M=160, EPS=1e-8, BM=4, GROUP=4, DOUBLING=0,
    )
    for h in (14, 28):
        for coarse in (0, 1):
            yield ("H2", h, coarse), _product_scan_coarse4_associative_backward_kernel, dict(
                height=h, width=h, modes=160, line_count=batch * h, packed_path_hidden=8,
                epsilon=1e-8, gain_kind=0, HAS_COARSE_GRAD=False, FULL_COARSE_GRAD=not coarse,
                COLLAPSE_PATHS_GRAD=True, PATH_SWIGLU=False, SWIGLU_HIDDEN=0, HAS_DESCRIPTOR_GRAD=False,
                PACKED_OUTPUT=False, COARSE_PATHS=bool(coarse), PLANAR_PATH_OUTPUT=False,
                COMBINED_STATE_GRADS=True, MEAN_DESCRIPTOR_GRAD=False, BATCH_PARTIAL_GRADS=True,
                ZERO_UNUSED_COARSE_GRADS=False, LINES_PER_GROUP=0 if coarse else 4,
                COMPACT_CHANNELS=0 if coarse else 3, BLOCK_HEIGHT=32 if coarse or h == 28 else 16,
                BLOCK_LINES=1, BLOCK_MODES=4, BLOCK_PATH_HIDDEN=16, BLOCK_SWIGLU_VALUE=16,
            )
    for h in (14, 28, 56):
        yield ("H4", h, 1), _product_scan_coarse4_associative_forward_kernel, dict(
            height=h, width=h, modes=160, line_count=batch * h, packed_path_hidden=8,
            epsilon=1e-8, gain_kind=0, EMIT_COARSE=False, FULL_COARSE=False,
            COLLAPSE_PATHS=True, PATH_SWIGLU=False, SWIGLU_HIDDEN=0, EMIT_DESCRIPTOR=False,
            BLOCK_HEIGHT=max(32, triton.next_power_of_2(h)), BLOCK_LINES=1, BLOCK_MODES=8,
            BLOCK_PATH_HIDDEN=16, BLOCK_SWIGLU_VALUE=16, PACKED_OUTPUT=False,
            COARSE_PATHS=True, PLANAR_PATH_OUTPUT=False, DETERMINISTIC_DESCRIPTOR=False,
        )
    for coarse in (0, 1):
        yield ("H5", 56, coarse, "backward"), path._horizontal_path_backward_kernel, dict(
            H=56, W=56, M=160, B=batch, STRIDE=320, BM=8, COARSE=bool(coarse), GROUP=1, DAR=None,
        )
    yield ("H5", 56, 0, "forward"), _bidirectional_forward_kernel, dict(
        height=56, width=56, modes=160, line_count=batch * 56, SOURCE_REAL_STRIDE=320,
        SOURCE_IMAG_STRIDE=320, BLOCK_WIDTH=64, BLOCK_MODES=8, COMBINED_STATES=True,
    )


def run(task):
    batch, index, config = task
    case, base, args = list(case_specs(batch))[index]
    newest("effective_output_vjp")
    label = f"{batch}-{'-'.join(map(str, case))}-{config}"
    result = dict(batch=batch, case=case, config=asdict(config))
    try:
        fn, grid = candidate_kernel(base, case, config, args, batch, 16)
        constexprs = {k: args[k] for k in fn.arg_names if k in args and args[k] is not None
                      and (k not in SCALARS or k in ("height",) and case[0] != "H5")}
        signature = {
            k: ("constexpr" if k in constexprs else "i32" if k in SCALARS
                else "*bf16" if k in BF16 or k.startswith(("source_", "grad_source_"))
                else "*fp32")
            for k in fn.arg_names
        }
        attrs = {(i,): [["tt.divisibility", 16]] for i, val in enumerate(signature.values())
                 if val.startswith("*")}
        # Bundled ptxas verifies the generated binary during compile.
        code = triton.compile(ASTSource(fn, signature=signature, constexprs=constexprs, attrs=attrs),
                              target=TARGET, options=dict(num_warps=config.warps, num_stages=config.stages))
        ptx = code.asm["ptx"]
        cubin = Path(code.metadata_group[next(k for k in code.metadata_group if k.endswith(".cubin"))])
        usage = subprocess.check_output([str(CUOBJDUMP), "--dump-resource-usage", str(cubin)], text=True)
        result.update(grid=grid, wgmma=ptx.count("wgmma.mma_async"), tma=ptx.count("cp.async.bulk.tensor"),
                      shared_bytes=code.metadata.shared,
                      **{k.lower(): int(v) for k, v in re.findall(r"(REG|STACK|LOCAL):(\d+)", usage)})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return label, result


def main():
    import lnet

    lnet.__path__.append(str(ROOT / "current/src/lnet"))
    assert triton.__version__ == "3.5.1"
    tasks = []
    for index, (case, _, args) in enumerate(case_specs(128)):
        family, h, coarse, *direction = case
        base = Config("r4", args.get("BM", args.get("BLOCK_MODES")), 4, group=args.get("GROUP", args.get("LINES_PER_GROUP", 0)))
        for conf in (base, *screen_configs(family, coarse, h, direction == ["forward"])):
            # Only baked-in TMA descriptors and hback B constants vary by batch.
            batches = (128, 256, 512, 1024) if conf.tma or direction == ["backward"] else (128,)
            tasks.extend((b, index, conf) for b in batches if b != 512 or family == "H3")
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for label, result in pool.map(run, tasks, chunksize=1):
            results[label] = result
            print(label, result.get("error", "compiled"), flush=True)
    save(ROOT / "offline-sm90.json", results)
    errors = sum("error" in row for row in results.values())
    print(f"SM90 preflight: {len(results) - errors}/{len(results)} compiled", flush=True)
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
