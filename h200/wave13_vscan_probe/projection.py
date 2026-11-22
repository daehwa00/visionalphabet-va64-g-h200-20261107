"""Measure individual planar path projection kernels with CUDA Graph replay."""

# This probe deliberately calls the private NVRTC launch API.
# ruff: noqa: SLF001, FBT001
import argparse
import ctypes as ct
import itertools
import json
import math
import statistics
import sys
import types
from pathlib import Path

import torch

from lnet import general_cuda_scans as scans

SHAPES = tuple(itertools.product((32, 128), (56, 28, 14), (False, True)))
VARIANTS = ("baseline", "occupancy", "vector", "vector-shared4")


def load_baseline(path: Path) -> types.ModuleType:
    module = types.ModuleType("lnet._vscan_projection_baseline")
    sys.modules[module.__name__] = module
    source = path.read_text(encoding="utf-8").split("def _reference_vertical(", 1)[0]
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def kernel_launch(
    module: types.ModuleType, variant: str, backward: bool, cells: int, modes: int, hidden: int,
) -> tuple[scans.Kernel, int, int, int]:
    tiles = math.ceil(modes / 32)
    fields = 11 * hidden + 2
    legacy = variant in {"baseline", "occupancy"}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    candidates = []
    for warps in ((2,) if variant == "baseline" else ((2, 4, 8) if backward else (4, 8))):
        if legacy:
            key = module.Kernel(
                torch.bfloat16, warps, backward, hidden8=True, cache_weights=True,
                projection=True, precise=not backward,
            )
            weights = 11 * hidden * 32 * 2
        else:
            key = scans.Kernel(
                torch.bfloat16, warps, backward, hidden8=True, cache_weights=True,
                projection=True, shared_acc=variant == "vector-shared4",
            )
            weights = (fields - 2 * backward) * 32 * 4
            if key.shared_acc:
                weights += (hidden // 2) * 8 * warps * 32 * 4
        smem = max(weights, fields * warps * 32 * 4 if backward else 0)
        if smem > scans.device_limits(0)[1]:
            continue
        resident = module._resident(0, key, smem)
        if not resident:
            continue
        if variant == "baseline":
            groups = min(math.ceil(cells / warps), math.ceil(4 * sms / tiles))
        else:
            groups = min(math.ceil(cells / (16 * warps)), math.ceil(4 * sms * resident / tiles))
        if backward:
            partial_bytes = fields * modes * 4
            groups = min(groups, max(1, 64 * 1024**2 // partial_bytes))
        score = min(sms * resident, groups * tiles) * warps
        candidates.append((score, key, groups, smem, resident))
    _, key, groups, smem, resident = max(candidates, key=lambda row: row[0])
    return key, groups, smem, resident


def replay_ms(graphs: dict, repeats: int) -> dict:
    samples = {label: [] for label in graphs}
    order = list(graphs)
    for iteration in range(max(5, len(order))):
        split = iteration % len(order)
        for label in order[split:] + order[:split]:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                graphs[label].replay()
            end.record()
            end.synchronize()
            samples[label].append(begin.elapsed_time(end) / repeats)
    return samples


def probe(
    batch: int, side: int, coarse: bool, modes: int, backward: bool, variants: list[str],
    baseline: types.ModuleType | None, repeats: int,
) -> dict:
    hidden = 8
    cells = batch * (side // (2 if coarse else 1)) ** 2
    features = torch.randn((8, cells, modes), device="cuda", dtype=torch.bfloat16)
    path = [
        torch.randn(shape, device="cuda").mul_(0.15)
        for shape in ((modes, hidden, 8), (modes, hidden), (modes, 2, hidden), (modes, 2))
    ]
    real = torch.randn_like(features[0])
    imag = torch.randn_like(real)
    adjoints = torch.empty_like(features) if backward else None
    args = (
        features, *path, real if backward else None, imag if backward else None,
        None if backward else real, None if backward else imag, adjoints,
    )
    tiles = math.ceil(modes / 32)
    graphs, partials, results = {}, {}, {}
    for variant in variants:
        if variant == "vector-shared4" and not backward:
            continue
        module = baseline if variant in {"baseline", "occupancy"} else scans
        key, groups, smem, resident = kernel_launch(
            module, variant, backward, cells, modes, hidden,
        )
        partial = features.new_empty(
            (groups, 11 * hidden + 2, modes), dtype=torch.float32,
        ) if backward else None
        partials[variant] = partial
        params = (
            *map(scans._ptr, (*args, partial)),
            ct.c_longlong(cells),
            *((ct.c_longlong(features.stride(0)),) if module is scans else ()),
            *map(ct.c_int, (modes, hidden, 0)),
        )
        module._launch_kernel(features, key, groups * tiles, smem, params)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            module._launch_kernel(features, key, groups * tiles, smem, params)
        graphs[variant] = graph
        results[variant] = {
            "warps": key.warps, "groups": groups, "resident": resident, "smem_bytes": smem,
        }
    for variant, samples in replay_ms(graphs, repeats).items():
        ms = statistics.median(samples)
        payload_bytes = features.element_size() * cells * modes * (18 if backward else 10)
        results[variant].update(ms=ms, gbps=payload_bytes / (ms * 1e6), samples=samples)
    return {
        "shape": [batch, side, side, modes, coarse], "backward": backward, "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    baseline = load_baseline(args.baseline_source) if args.baseline_source else None
    if baseline is None and {"baseline", "occupancy"}.intersection(args.variants):
        parser.error("--baseline-source is required for baseline and occupancy")
    torch.manual_seed(731)
    cases = ((32, 56, False), (128, 56, False)) if args.quick else SHAPES
    with args.out.open("w", encoding="utf-8") as out:
        for (batch, side, coarse), backward in itertools.product(cases, (False, True)):
            row = probe(batch, side, coarse, 160, backward, args.variants, baseline, args.repeats)
            print(json.dumps(row), file=out, flush=True)


if __name__ == "__main__":
    main()
