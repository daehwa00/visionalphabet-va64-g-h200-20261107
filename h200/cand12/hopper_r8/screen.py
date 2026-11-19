"""Screen checkpoint operands in the isolated compiled r4 workload process."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import torch
import triton

from run_all import ROOT, save
from variants import hback_mode_tail, load_module, source_for

PROOF = ROOT / "proof"
ATOMIC = ("grad_path_", "grad_decay_", "grad_gamma_")
BUNDLED_CUOBJDUMP = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
CUOBJDUMP = shutil.which("cuobjdump") or (
    str(BUNDLED_CUOBJDUMP) if BUNDLED_CUOBJDUMP.is_file() else None
)


@dataclass(frozen=True)
class Config:
    variant: str
    bm: int
    warps: int
    stages: int = 1
    lines: int = 1
    group: int = 1
    scheduler: str = "grid"
    dot: str = ""
    tma: bool = False
    bh: int = 0


def fields(fn, args, kwargs):
    signature = fn.triton_meta["signature"]
    constants = {**fn.triton_meta.get("constants", {}), **fn.launchers[0].config.kwargs}
    names = [name for name, kind in signature.items() if kind != "constexpr" and name not in constants]
    if len(args) not in (len(names), len(names) + 3):
        raise ValueError(f"Inductor signature mismatch for {fn.fn.__name__}: {len(args)=}, {names=}")
    geometry = fn.launchers[0].config
    grid = tuple(map(int, args[len(names):]))
    if not grid:
        for record in fn.inductor_meta.get("precomputed_grids", ()):
            if record["config"] == dict(geometry.kwargs, num_warps=geometry.num_warps,
                                        num_stages=geometry.num_stages):
                grid = tuple(record["python"])
                break
    known = {**constants, **dict(zip(names, args)), **{k: v for k, v in kwargs.items() if k in names}}
    return known, grid, geometry


def describe(name, args):
    h = args.get("height", args.get("H"))
    coarse = args.get("COARSE", args.get("COARSE_PATHS", False))
    if h == 56 and name == "_path_backward_kernel":
        return "H1", h, int(coarse)
    if (h in (14, 28) and name == "_product_scan_coarse4_associative_backward_kernel"
            and args.get("COLLAPSE_PATHS_GRAD") and args.get("BATCH_PARTIAL_GRADS")
            and args.get("COMBINED_STATE_GRADS") and not args.get("HAS_COARSE_GRAD")
            and not args.get("HAS_DESCRIPTOR_GRAD") and args.get("packed_path_hidden") == 8
            and bool(args.get("FULL_COARSE_GRAD")) != bool(coarse)):
        return "H2", h, int(coarse)
    if h == 56 and name == "_path_forward_kernel":
        return "H3", h, int(coarse)
    if (h in (14, 28, 56) and name == "_product_scan_coarse4_associative_forward_kernel" and coarse
            and args.get("COLLAPSE_PATHS") and not args.get("EMIT_COARSE")
            and not args.get("EMIT_DESCRIPTOR") and args.get("packed_path_hidden") == 8):
        return "H4", h, 1
    if h == 56 and name == "_horizontal_path_backward_kernel":
        return "H5", h, int(coarse), "backward"
    if h == 56 and name == "_bidirectional_forward_kernel" and args.get("COMBINED_STATES"):
        return "H5", h, 0, "forward"
    return None


def outputs(case, args):
    family, _, *rest = case
    named = list(args)
    if family == "H3":
        return ["OR", "OI"], []
    if family == "H4":
        out, atomic = ["collapsed_real", "collapsed_imag"], []
        if args["EMIT_COARSE"]:
            out.extend(("coarse_real", "coarse_imag"))
        if args["EMIT_DESCRIPTOR"]:
            out.append("descriptor_energy")
            atomic.append("descriptor_energy")
        return out, atomic
    if family == "H5" and rest[-1] == "forward":
        return ["positive_real"], []
    if "DS" in args:
        atomic = ["DAR", "DAI", "DGR", "DGI"]
        return [*atomic, "DS"], atomic
    atomic = [n for n in named if n.startswith(ATOMIC)]
    out = ["grad_source_real_a"] if family in ("H1", "H2") else ["grad_source_real"]
    if family == "H2" and not args.get("COMBINED_STATE_GRADS"):
        out.extend(("grad_source_imag_a", "grad_source_real_b", "grad_source_imag_b"))
    if family == "H5" and not args.get("PACKED_SOURCE_GRADIENT"):
        out.append("grad_source_imag")
    return [*atomic, *out], atomic


def output_buffers(args, names):
    """Keep output storage aliasing and alignment and replay without mutating training."""
    raw_buffers = {}
    new_args = args.copy()
    for name in names:
        storage = args[name].untyped_storage()
        if storage._cdata not in raw_buffers:
            raw_buffers[storage._cdata] = torch.empty(storage.nbytes(), device="cuda", dtype=torch.uint8)
    for name, tensor in args.items():
        if torch.is_tensor(tensor):
            raw = raw_buffers.get(tensor.untyped_storage()._cdata)
            if raw is not None:
                new_args[name] = torch.empty(0, device=tensor.device, dtype=tensor.dtype).set_(
                    raw.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride()
                )
                assert tensor.data_ptr() % 16 == new_args[name].data_ptr() % 16
    return new_args


def compare(value, expected):
    max_abs = ref_peak = diff_sq = ref_sq = 0.0
    chunks = zip(value.reshape(-1).split(1 << 22), expected.reshape(-1).split(1 << 22), strict=True)
    for actual, ref in chunks:
        actual, ref = actual.float(), ref.float()
        delta = actual - ref
        d, r, top, peak = torch.stack(
            (delta.square().sum(), ref.square().sum(), delta.abs().max(), ref.abs().max())
        ).tolist()
        if not all(map(math.isfinite, (d, r, top, peak))):
            return dict(finite=False, max_abs=None, rel_l2=None, passed=False)
        diff_sq += d
        ref_sq += r
        max_abs = max(max_abs, top)
        ref_peak = max(ref_peak, peak)
    rel = math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-12)
    return dict(finite=True, max_abs=max_abs, rel_l2=rel,
                passed=rel <= .005 and max_abs <= .02 * ref_peak + 1e-6)



def compare_output(name, actual, expected, case, args, batch):
    family, h, coarse, *_ = case
    if name != "grad_source_real_a" or not coarse or family not in ("H1", "H2"):
        return compare(actual, expected)
    if family == "H2" and args["ZERO_UNUSED_COARSE_GRADS"]:
        return compare(actual, expected)
    # Retained coarse gradients only define the odd columns in A and even
    # columns in B. The downstream horizontal backward uses the same mask.
    shape = (4, batch, h, h, args["modes"])
    refs = zip(actual.view(shape), expected.view(shape), strict=True)
    metrics = [compare(x[..., 1 - i // 2::2, :], y[..., 1 - i // 2::2, :])
               for i, (x, y) in enumerate(refs)]
    finite = all(row["finite"] for row in metrics)
    return dict(finite=finite, passed=all(row["passed"] for row in metrics),
                max_abs=max(row["max_abs"] for row in metrics) if finite else None,
                rel_l2=max(row["rel_l2"] for row in metrics) if finite else None)


def compile_proof(compiled):
    ptx, ttir = compiled.asm["ptx"], compiled.asm["ttir"]
    ptx_hash, ttir_hash = (hashlib.sha256(s.encode()).hexdigest() for s in (ptx, ttir))
    stem = PROOF / ptx_hash[:20]
    stem.with_suffix(".ptx").write_text(ptx)
    stem.with_suffix(".ttir").write_text(ttir)
    metadata = compiled.metadata._asdict()
    stem.with_suffix(".json").write_text(str(metadata))
    result = dict(wgmma_count=ptx.count("wgmma.mma_async"), tma_count=ptx.count("cp.async.bulk.tensor"),
                  n_regs=compiled.n_regs, n_spills=compiled.n_spills,
                  shared_bytes=metadata["shared"], ptx_sha256=ptx_hash, ttir_sha256=ttir_hash,
                  ttir_dots=ttir.count("tt.dot"), ttir_descriptor_ops=ttir.count("tt.descriptor_"),
                  local_declarations=re.findall(r"\.local [^;]+", ptx)[:3],
                  local_loads=ptx.count("ld.local"), local_stores=ptx.count("st.local"),
                  stack_bytes=None, local_bytes=None)
    if CUOBJDUMP:
        binfile = stem.with_suffix(".cubin")
        binfile.write_bytes(compiled.asm["cubin"])
        proc = subprocess.run([CUOBJDUMP, "--dump-resource-usage", binfile], text=True,
                              capture_output=True, timeout=10, check=False)
        for opname, key in (("STACK", "stack_bytes"), ("LOCAL", "local_bytes")):
            found = re.search(rf"\b{opname}:(\d+)", proc.stdout)
            if found:
                result[key] = int(found[1])
    return result


def screen_configs(family, coarse, h, horizontal_forward=False):
    """Keep the mechanism and pipeline-stage sweeps on matched geometries."""
    if family in ("H1", "H3"):
        variant = "pack8" if family == "H1" else "vfwd2out"
        yield Config(variant, 8, 4, group=4)
        yield Config(variant, 8, 4, group=4, tma=True)
        yield Config(variant, 8, 4, group=4, dot="wgmma")
        if family == "H3":
            yield Config(variant, 8, 4, group=4, dot="wgmma", tma=True)
        for stage in (2, 3, 4):
            yield Config(variant, 8, 4, stage, group=4)
        yield Config(variant, 16, 4, group=2)
        yield Config(variant, 8, 4, group=4, scheduler="persistent")
        if family == "H1" and coarse:
            yield Config("parity", 8, 4, group=4)
    elif family == "H2":
        if coarse:
            group = 4 if h == 14 else 1
            yield Config("r6", 8, 2, group=group)
            yield Config("r6", 8, 4, 2, group=group)
            yield Config("r6", 16, 4, 3, group=4)
            yield Config("r6", 8, 4, 4, lines=2, group=1)
        else:
            yield Config("r5", 4 if h == 14 else 8, 2 if h == 14 else 4, group=4)
            yield Config("r7", 4, 2, lines=2, group=4)
            yield Config("r7", 4, 4, 2, lines=2, group=4)
            yield Config("r7", 8, 4, 3, lines=2, group=4)
            yield Config("r7", 8, 4, 4, lines=4, group=4)
    elif family == "H4":
        from lnet.r7_dispatch import FORWARD_GEOMETRY

        bh, bm, lines, warps, _ = FORWARD_GEOMETRY[h, True]
        yield Config("r7", bm, warps, lines=lines, bh=bh)
        yield Config("r7", 8, 4, 2, lines=lines, bh=bh)
        yield Config("r7", 16, 4, 3, lines=lines, bh=bh)
        yield Config("r7", 8, 8, 4, lines=lines, bh=max(32, bh))
        yield Config("r7", 8, 4, lines=64 // bh, dot="wgmma", bh=bh)
        if h == 56:
            yield Config("seg4pack2", 8, 4, bh=64)
    else:
        for bm, warps in ((32, 4), (64, 8)):
            for stage in (1, 2, 3, 4):
                yield Config("seg4", bm, warps, stage)
        if horizontal_forward:
            yield Config("r4", 32, 4, 2)


_LOADED = {}


def newest(module):
    if module not in _LOADED:
        src = ROOT / "current/src/lnet" / f"{module}.py"
        if module in ("r6_associative", "r7_associative", "r7_direct"):
            target = PROOF / "generated" / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(src.read_text().replace(
                "from .effective_output_vjp import ", "from .hopper_r8_effective_output_vjp import "
            ))
            src = target
        _LOADED[module] = load_module(f"lnet.hopper_r8_{module}", src)
    return _LOADED[module]


def candidate_kernel(base_fn, case, config, args, batch, sm_count):
    from associative_experiments import associative_forward
    from horizontal_experiments import bifwd_seg4
    from lnet.r6_horizontal import hback56
    from parity_experiment import c1_parity32
    from lnet.r7_dispatch import FORWARD_GEOMETRY

    family, h, coarse, *direction = case
    b = config
    modes = args.get("modes", args.get("M"))
    mode_tiles = triton.cdiv(modes, b.bm)
    fn = base_fn
    if b.variant == "pack8":
        fn = newest("effective_structural_scan")._BACKWARD_VARIANTS[2]
    elif b.variant == "vfwd2out":
        fn = newest("effective_structural_scan")._FORWARD_VARIANTS[2]
    elif b.variant == "parity":
        fn = c1_parity32
    elif b.variant == "r5":
        fn = newest("effective_output_vjp")._assoc_fast_kernel
    elif b.variant == "r6":
        fn = newest("r6_associative").coarse_associative_backward
        args["BLOCK_HEIGHT"] = triton.next_power_of_2(h)
    elif b.variant == "r7" and family == "H2":
        fn = newest("r7_direct").r7_direct_backward
        args.update(BLOCK_HEIGHT=triton.next_power_of_2(h), COMPACT_CHANNELS=3)
    elif b.variant == "r7" and family == "H4":
        fn = newest("r7_associative").r7_associative_forward
        args.update(BLOCK_HEIGHT=b.bh, SINGLE_LOAD=FORWARD_GEOMETRY[h, bool(coarse)][-1])
    elif b.variant == "seg4pack2":
        fn = associative_forward
        args.update(PACK2=True, FAST_SIGMOID=False, SEGMENT=True, BLOCK_HEIGHT=b.bh)
    elif b.variant == "seg4" and direction == ["forward"]:
        fn = bifwd_seg4
        args["DUAL"] = False
    elif b.variant == "seg4":
        fn = hback56 if modes % b.bm == 0 else hback_mode_tail(hback56, PROOF / "generated")
        args.update(SEGMENT=True, PARTIAL=False, partials=args["DAR"])

    if family in ("H1", "H3"):
        if modes % b.bm:
            raise ValueError("Vertical mode tile must divide the mode count")
        lines = batch * h // (2 if coarse else 1)
        grid = (mode_tiles * b.group, lines // b.group)
        args.update(BM=b.bm, GROUP=b.group)
        line_axis = 1
    elif family in ("H2", "H4"):
        tiles = batch * triton.cdiv(args["width"] // (2 if coarse else 1), b.lines)
        group = b.group if family == "H2" else 0
        args.update(BLOCK_MODES=b.bm, BLOCK_LINES=b.lines)
        if family == "H2":
            args["LINES_PER_GROUP"] = group
        grid = (mode_tiles * group, tiles // group) if group else (tiles, mode_tiles)
        line_axis = int(bool(group))
    elif "BM" in args:
        args["BM"] = b.bm
        grid, line_axis = (mode_tiles, batch * h), 1
    else:
        args["BLOCK_MODES"] = b.bm
        grid, line_axis = (args["line_count"], mode_tiles), 0

    if b.scheduler != "grid" or b.dot or b.tma:
        fn = source_for(fn, PROOF / "generated", scheduler=b.scheduler, line_axis=line_axis,
                        dot=b.dot, tma=b.tma, forward=family == "H3")
    if b.tma:
        args["TMA_BATCH"] = batch
    if b.scheduler != "grid":
        args.update(GRID0=grid[0], GRID1=grid[1], SERIAL_LINES=b.lines)
        grid = (triton.cdiv(math.prod(grid), b.lines) if b.scheduler == "serial" else sm_count,)
    return fn, grid


def make_launch(fn, args, grid, config):
    params = {name: args[name] for name in fn.arg_names if name in args}
    return lambda: fn[grid](**params, num_warps=config.warps, num_stages=config.stages)


def benchmark(launch, reset, flush, reps=5):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(reps):
        reset()
        flush.add_(1)
        begin.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(round(1000 * begin.elapsed_time(end), 2))
    return samples


def replay(fn, case, reference, grid, geometry, row, checkpoint, deadline, batch, sm_count, pilot):
    h = case[1]
    source = next(reference[n] for n in ("source_real_a", "SA", "SR", "source_real") if n in reference)
    assert source.dtype == torch.bfloat16 and source.numel() >= batch * h * h
    if source.ndim >= 4:
        assert source.shape[0] == batch, source.shape
    out, atomics = outputs(case, reference)
    working = output_buffers(reference, out)

    def reset(poison=False):
        for name in atomics:
            working[name].zero_()
        if poison:
            for name in out:
                if name not in atomics:
                    working[name].fill_(math.nan)

    base = Config("r4", reference.get("BM", reference.get("BLOCK_MODES")),
                  geometry.num_warps, geometry.num_stages, reference.get("BLOCK_LINES", 1),
                  reference.get("GROUP", reference.get("LINES_PER_GROUP", 0)))
    if len(grid) == 3:
        grid = tuple(grid)
    elif not grid:
        _, grid = candidate_kernel(fn, case, base, working.copy(), batch, sm_count)
    run_base = make_launch(fn, working, grid, base)
    row.update(shape=[batch, h, h, reference.get("modes", reference.get("M"))], dtype="bfloat16",
               source_stride=list(source.stride()), grid=grid, base=asdict(base), trials=[])
    reset(poison=True)
    compiled = run_base()
    torch.cuda.synchronize()
    row["baseline_code"] = compile_proof(compiled)
    row["baseline_vs_live"] = {n: compare_output(n, working[n], reference[n], case, reference, batch) for n in out}
    if not all(item["passed"] for item in row["baseline_vs_live"].values()):
        row["status"] = "baseline_numeric_failure"
        return
    flush = torch.empty(128 << 20, device="cuda", dtype=torch.uint8).zero_()
    baseline = benchmark(run_base, reset, flush, reps=7)
    row["baseline_us"] = baseline
    row["baseline_median_us"] = statistics.median(baseline)
    checkpoint()
    start = time.monotonic()
    seen = {base}
    options = list(screen_configs(case[0], case[2], h, case[-1] == "forward"))
    if pilot:
        options = options[:4] if case[0] == "H3" else options[:3]
    for config in options:
        if config in seen:
            continue
        limit = (100 if case[0] in ("H1", "H3") else 72) if batch == 1024 else (60 if case[0] in ("H1", "H3") else 42)
        if time.time() >= deadline or time.monotonic() - start > limit:
            row["status"] = "budget_limited"
            return
        seen.add(config)
        trial = dict(config=asdict(config))
        row["trials"].append(trial)
        print("SCREEN_TRIAL", case, asdict(config), flush=True)
        try:
            arg_map = working.copy()
            candidate, candidate_grid = candidate_kernel(fn, case, config, arg_map, batch, sm_count)
            run = make_launch(candidate, arg_map, candidate_grid, config)
            reset(poison=True)
            compiled = run()
            torch.cuda.synchronize()
            trial.update(grid=candidate_grid, code=compile_proof(compiled))
            numeric = {n: compare_output(n, working[n], reference[n], case, reference, batch) for n in out}
            trial["numeric_pass"] = all(item["passed"] for item in numeric.values())
            trial["max_abs"] = max((item["max_abs"] for item in numeric.values()
                                    if item["max_abs"] is not None), default=None)
            trial["rel_l2"] = max((item["rel_l2"] for item in numeric.values()
                                   if item["rel_l2"] is not None), default=None)
            trial["failed_outputs"] = [n for n, v in numeric.items() if not v["passed"]]
            run_times = benchmark(run, reset, flush)
            trial["samples_us"] = run_times
            trial["warm_median_us"] = statistics.median(run_times)
            baseline = benchmark(run_base, reset, flush, reps=3)
            trial["baseline_median_us"] = statistics.median(baseline)
            trial["speedup"] = round(trial["baseline_median_us"] / trial["warm_median_us"], 6)
        except Exception as exc:
            trial["error"] = f"{type(exc).__name__}: {str(exc)[-380:]}"
        checkpoint()
    row["status"] = "complete"


def install(batch, *, pilot=False):
    import lnet
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    from run_all import gpu_topology

    # Import candidate source beside the frozen baseline modules. The MLP
    # candidates need the current VJP helpers even when replay runs from r4.
    lnet.__path__.append(str(ROOT / "current/src/lnet"))
    PROOF.mkdir(exist_ok=True)
    newest("effective_output_vjp")
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    deadline = int(os.environ["HOPPER_SCREEN_DEADLINE"])
    cases, counts, missing = {}, Counter(), Counter()
    topology = gpu_topology()
    report = dict(cases=cases, capture_counts=counts, capture_errors=missing, microbatch=batch,
                  epoch=18, global_step=18270, triton=triton.__version__, torch=torch.__version__,
                  gpu_topology=topology, l2_flush_bytes=128 << 20, rel_l2_limit=.005, peak_abs_scale=.02)

    def checkpoint():
        save(ROOT / f"screen-{batch}.json", report)

    original = CachingAutotuner.run
    prefixes = {"_path_backward_kernel", "_path_forward_kernel", "_product_scan_coarse4_associative_backward_kernel",
                "_product_scan_coarse4_associative_forward_kernel", "_horizontal_path_backward_kernel",
                "_bidirectional_backward_kernel", "_bidirectional_forward_kernel"}

    def run(operator, *args, **kwargs):
        output = original(operator, *args, **kwargs)
        if operator.fn.__name__ not in prefixes or kwargs.get("benchmark_run") or time.time() >= deadline:
            return output
        try:
            named, grid, geometry = fields(operator, args, kwargs)
            case = describe(operator.fn.__name__, named)
        except Exception as exc:
            missing[f"{operator.fn.__name__}: {str(exc)[-220:]}"] += 1
            checkpoint()
            return output
        if case is None or (pilot and case[0] != "H3"):
            return output
        tag = "-".join(map(str, case))
        counts[tag] += 1
        if tag in cases:
            return output
        row = cases[tag] = dict(status="captured", kernel=operator.fn.__name__)
        checkpoint()
        try:
            with torch.no_grad():
                replay(operator.fn, case, named, grid, geometry, row, checkpoint, deadline,
                       batch, topology["multi_processor_count"], pilot)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {str(exc)[-500:]}")
        print("SCREEN_CASE", tag, row["status"], flush=True)
        checkpoint()
        return output

    CachingAutotuner.run = run
    checkpoint()
