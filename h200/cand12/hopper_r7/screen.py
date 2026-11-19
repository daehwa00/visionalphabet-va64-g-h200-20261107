"""Screen checkpoint operands in the isolated compiled r4 workload process."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import time

import torch
import triton

from run_all import ROOT, save
from variants import load_module, source_for

PROOF = ROOT / "proof"
ATOMIC = ("grad_path_", "grad_decay_", "grad_gamma_")
CUOBJDUMP = shutil.which("cuobjdump") or shutil.which("cuobjdump", path="/usr/local/cuda/bin")


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
    if h not in (14, 28, 56):
        return None
    coarse = args.get("COARSE", args.get("COARSE_PATHS", args.get("COARSE_EVEN_ODD", False)))
    if name == "_path_backward_kernel":
        return "H1", h, int(coarse)
    if name == "_product_scan_coarse4_associative_backward_kernel" and args.get("COLLAPSE_PATHS_GRAD"):
        return "H2", h, int(coarse)
    if name == "_path_forward_kernel":
        return "H3", h, int(coarse)
    if name == "_product_scan_coarse4_associative_forward_kernel" and args.get("COLLAPSE_PATHS"):
        return "H4", h, int(coarse)
    if name in ("_horizontal_path_backward_kernel", "_bidirectional_backward_kernel"):
        return "H5", h, int(coarse), "backward"
    if name == "_bidirectional_forward_kernel" and args.get("COMBINED_STATES"):
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


def compile_proof(compiled):
    ptx, ttir = compiled.asm["ptx"], compiled.asm["ttir"]
    ptx_hash, ttir_hash = (hashlib.sha256(s.encode()).hexdigest() for s in (ptx, ttir))
    stem = PROOF / ptx_hash[:20]
    stem.with_suffix(".ptx").write_text(ptx)
    stem.with_suffix(".ttir").write_text(ttir)
    metadata = compiled.metadata._asdict()
    stem.with_suffix(".json").write_text(str(metadata))
    result = dict(has_wgmma="wgmma.mma_async" in ptx, has_tma="cp.async.bulk.tensor" in ptx,
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


def screen_configs(family, coarse, h, base, horizontal_forward=False):
    """Fractional geometry matrix followed by a few isolated mechanism tests."""
    if family in ("H1", "H3"):
        variant = ("parity" if coarse else "pack8") if family == "H1" else "vfwd2out"
        data = ((4, 2, 1, 1, 1), (4, 4, 1, 1, 4), (8, 4, 1, 1, 4),
                (8, 8, 1, 1, 8), (16, 4, 1, 1, 2), (32, 8, 1, 1, 1),
                (8, 4, 2, 1, 2), (16, 8, 3, 1, 4), (4, 4, 4, 1, 1))
        yield Config(variant, base.bm, base.warps, 1, 1, base.group)
        yield Config(variant, 8, 4, dot="wgmma", group=4)
        yield Config(variant, 8, 4, tma=True, group=4)
        yield Config(variant, 8, 4, group=4, scheduler="persistent")
        yield Config(variant, 8, 4, 2, 1, 4, "warp_specialize", tma=True)
        yield Config(variant, 8, 4, 1, 2, 4, "serial")
        yield Config(variant, 16, 8, 2, 4, 2, "serial")
        for bm, w, stages, lines, group in data:
            if family == "H3" and stages == 4:
                continue
            yield Config(variant, bm, w, stages, lines, group)
            if stages == 1:
                yield Config("r4", bm, w, stages, lines, group)
    elif family in ("H2", "H4"):
        variant = ("r6" if coarse else "r5") if family == "H2" else "pack2"
        data = ((8, 2, 1, 1, 4), (8, 4, 1, 2, 1), (16, 4, 2, 1, 4),
                (16, 8, 1, 4, 2), (32, 8, 3, 1, 1), (8, 4, 2, 4, 4),
                (16, 4, 1, 1, 4), (8, 8, 1, 8, 1))
        yield Config(variant, base.bm, base.warps, group=base.group)
        # Flattened lines are part of the dot's M dimension.
        yield Config(variant, 8, 4, lines=triton.cdiv(64, triton.next_power_of_2(h)), dot="wgmma")
        if family == "H4":
            if h == 56 and coarse:
                yield Config("seg4pack2", 8, 4)
                yield Config("seg4pack2", 16, 4, 2, 2)
            yield Config("pack2", 8, 4, 2, 1, 0, dot="ieee")
        else:
            yield Config(variant, 8, 4, group=4, scheduler="persistent")
        for bm, w, stages, lines, group in data:
            if family == "H4" and lines == 8:
                continue
            yield Config(variant, bm, w, stages, lines, group if family == "H2" else 0)
    else:
        variant = "seg4" if h == 56 else "r4"
        for bm, w, stages in ((16, 4, 1), (32, 4, 1), (32, 8, 2), (64, 8, 3), (16, 8, 3)):
            yield Config(variant, bm, w, stages)
        yield Config(variant, 16, 4, scheduler="persistent")
        yield Config(variant, 16, 8, 2, 2, 1, "serial")


def candidate_kernel(base_fn, case, config, args):
    newest("effective_output_vjp")
    from associative_experiments import associative_forward
    from horizontal_experiments import bifwd_seg4
    from lnet.r6_horizontal import hback56
    from parity_experiment import c1_parity32

    family, h, coarse, *direction = case
    mode_count = args.get("modes", args.get("M"))
    b = config
    tile_count = triton.cdiv(mode_count, b.bm)
    fn = base_fn
    if b.variant in ("pack8", "vfwd2out", "r5"):
        fn = newest("effective_output_vjp")._assoc_fast_kernel if b.variant == "r5" else (
            newest("effective_structural_scan")._FORWARD_VARIANTS[2] if b.variant == "vfwd2out"
            else newest("effective_structural_scan")._BACKWARD_VARIANTS[2])
    elif b.variant == "parity":
        fn = c1_parity32
    elif b.variant == "r6":
        fn = newest("r6_associative").coarse_associative_backward
    elif b.variant in ("pack2", "seg4pack2"):
        fn = associative_forward
        args.update(PACK2=True, FAST_SIGMOID=False, SEGMENT=b.variant == "seg4pack2")
    elif b.variant == "seg4" and direction == ["forward"]:
        fn = bifwd_seg4
        args["DUAL"] = False
    elif b.variant == "seg4":
        fn = hback56
        args.update(SEGMENT=True, PARTIAL=False, partials=args["DAR"])

    if family in ("H1", "H3"):
        if mode_count % b.bm:
            raise ValueError("Retained vertical mode tile must divide the mode count")
        line_count = 1024 * h // (2 if coarse else 1)
        grid = (tile_count * b.group, line_count // b.group)
        args.update(BM=b.bm, GROUP=b.group)
        line_axis = 1
    elif family in ("H2", "H4"):
        tiles = 1024 * triton.cdiv(args["width"] // (2 if coarse else 1), b.lines)
        group = b.group if family == "H2" else 0
        args.update(BLOCK_MODES=b.bm, BLOCK_LINES=b.lines)
        if family == "H2":
            args["LINES_PER_GROUP"] = group
        if b.variant == "r6" and coarse and h == 14:
            args["BLOCK_HEIGHT"] = 16
        grid = (tile_count * group, tiles // group) if group else (tiles, tile_count)
        line_axis = int(bool(group))
    elif "BM" in args:
        args["BM"] = b.bm
        grid, line_axis = (tile_count, 1024 * h), 1
    else:
        args["BLOCK_MODES"] = b.bm
        grid, line_axis = (1024 * h, tile_count), 0

    if b.scheduler != "grid" or b.dot or b.tma:
        fn = source_for(fn, PROOF / "generated", scheduler=b.scheduler, line_axis=line_axis,
                        dot=b.dot, tma=b.tma, forward=family == "H3")
    if b.scheduler != "grid":
        args.update(GRID0=grid[0], GRID1=grid[1], SERIAL_LINES=b.lines)
        count = math.prod(grid)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        grid = (triton.cdiv(count, b.lines) if b.scheduler == "serial" else sms,)
    return fn, grid


_LOADED = {}


def newest(module):
    if module not in _LOADED:
        src = ROOT / "current/src/lnet" / f"{module}.py"
        if module == "r6_associative":
            target = PROOF / "generated" / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(src.read_text().replace(
                "from .effective_output_vjp import ", "from .hopper_r7_effective_output_vjp import "
            ))
            src = target
        _LOADED[module] = load_module(f"lnet.hopper_r7_{module}", src)
    return _LOADED[module]


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


def replay(fn, case, reference, grid, geometry, row, checkpoint, deadline):
    h = case[1]
    assert reference.get("B", reference.get("line_count", 1024 * h) // h) == 1024
    source = next(reference[n] for n in ("source_real_a", "SA", "SR", "source_real") if n in reference)
    assert source.dtype == torch.bfloat16 and source.numel() > 0
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
        _, grid = candidate_kernel(fn, case, base, working.copy())
    run_base = make_launch(fn, working, grid, base)
    row.update(shape=[1024, h, h, reference.get("modes", reference.get("M"))], dtype="bfloat16",
               source_stride=list(source.stride()), grid=grid, base=asdict(base), trials=[])
    reset(poison=True)
    compiled = run_base()
    torch.cuda.synchronize()
    row["baseline_code"] = compile_proof(compiled)
    row["baseline_vs_live"] = {n: compare(working[n], reference[n]) for n in out}
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
    cap = torch.cuda.get_device_capability()[0]
    options = screen_configs(case[0], case[2], h, base, case[-1] == "forward")
    for config in options:
        if config in seen or (config.tma and cap < 9):
            continue
        limit = 65 if case[0] == "H1" else 36
        if time.time() >= deadline or time.monotonic() - start > limit:
            row["status"] = "budget_limited"
            return
        seen.add(config)
        trial = dict(config=asdict(config))
        row["trials"].append(trial)
        print("SCREEN_TRIAL", case, asdict(config), flush=True)
        if case[0] == "H3" and config.scheduler == "warp_specialize":
            trial["error"] = "Offline SM90 Triton 3.5.1 rejected this TMA loop in NVGPUWarpSpecialization task propagation"
            checkpoint()
            continue
        try:
            arg_map = working.copy()
            candidate, candidate_grid = candidate_kernel(fn, case, config, arg_map)
            run = make_launch(candidate, arg_map, candidate_grid, config)
            reset(poison=True)
            compiled = run()
            torch.cuda.synchronize()
            trial.update(grid=candidate_grid, code=compile_proof(compiled))
            numeric = {n: compare(working[n], reference[n]) for n in out}
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


def install():
    import lnet
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    # Keep r4 source resolution for the measured model. Only new r6 modules are added.
    lnet.__path__.append(str(ROOT / "current/src/lnet"))
    PROOF.mkdir(exist_ok=True)
    deadline = int(os.environ["HOPPER_SCREEN_DEADLINE"])
    cases, counts, missing = {}, Counter(), Counter()
    report = dict(cases=cases, capture_counts=counts, capture_errors=missing,
                  triton=triton.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()),
                  multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,
                  l2_flush_bytes=128 << 20, rel_l2_limit=.005, peak_abs_scale=.02)

    def checkpoint():
        save(ROOT / "screen.json", report)

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
        if case is None:
            return output
        tag = "-".join(map(str, case))
        counts[tag] += 1
        if tag in cases:
            return output
        row = cases[tag] = dict(status="captured", kernel=operator.fn.__name__)
        checkpoint()
        try:
            with torch.no_grad():
                replay(operator.fn, case, named, grid, geometry, row, checkpoint, deadline)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {str(exc)[-500:]}")
        print("SCREEN_CASE", tag, row["status"], flush=True)
        checkpoint()
        return output

    CachingAutotuner.run = run
    checkpoint()
