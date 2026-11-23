"""Run the integrated benchmark with Wave 14B launches substituted inside the custom op."""

from __future__ import annotations

# The prototype routes calls through the existing private CUDA launch API.
# ruff: noqa: SLF001
import argparse
import json
from collections import Counter
from functools import cache
from pathlib import Path

from probe import DIV, DTYPE, TYPES, Launch, compile_probe, write_json
from sourcegen import RING, SPATIAL, WEIGHT, WEIGHT_STREAM

from h200.liverms_v2_opt import bench_h200 as bench
from lnet import factorized_reader_kernels as reader

TILES = {tile.name: tile for tile in (*RING, *SPATIAL, *WEIGHT)}
VARIANT = "17-p1-nr4-1024d-v3"
POLICY_OPS = ("forward", "dinput", "dweight")


def select_policy(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))["shapes"]
    policy = {}
    for shape, row in rows.items():
        _, h, w, modes, rank = map(int, shape.split("x"))
        if rank != 4:
            continue
        key = f"{h}x{w}x{modes}x{rank}"
        policy[key] = {}
        for op in POLICY_OPS:

            def cost(name: str, row: dict = row, op: str = op) -> float:
                compute = row[op][name]["median_ms"]
                if op == "dweight":
                    compute += row["dweight_reduce"][name]["median_ms"]
                return compute

            valid = [
                name
                for name, timing in row[op].items()
                if name != "wave13"
                and timing["wave13_parity"]["outside_tolerance"] == 0
                and timing["pytorch_parity"]["outside_tolerance"] == 0
            ]
            if valid:
                policy[key][op] = min(valid, key=cost)
    return policy


@cache
def prepare(shape: tuple[int, int, int, int, int], op: str, name: str) -> tuple:
    b, h, w, m, rank = shape
    if name in WEIGHT_STREAM:
        ranks, rows, warps, vec, _ = WEIGHT_STREAM[name]
        module = reader._reader_cuda_module(
            0,
            (b, h, w, m),
            reader._reader_columns(w),
            rank,
            3,
            TYPES,
            strict=False,
            weight_geometry=(ranks, rows, warps),
            weight_vec=vec,
        )
        return module, "reader_dweight", 0
    tile = TILES[name]
    module = compile_probe(shape, {name})
    if tile in WEIGHT:
        smem = tile.weight_smem(rank)
    elif tile in SPATIAL:
        smem = tile.spatial_smem(rank, backward=op == "dinput")
    else:
        smem = 0
    kernel = f"reader_{tile.name}_{op}"
    module.load_function(kernel, smem)
    return module, kernel, smem


def install(policy: dict, counts: Counter) -> None:
    original = reader._launch_reader_cuda

    def route(module, op, grid, tensors, sizes, *, stream, threads=128):  # noqa: ANN001, ANN202
        if op in POLICY_OPS:
            h, w, m, rank = (*sizes[:3], sizes[-1])
            b = tensors[0].shape[0]
            shape = (b, h, w, m, rank)
            key = f"{h}x{w}x{m}x{rank}"
            choice = policy.get(key, {}).get(op)
            # Real packed BF16 with aligned component planes only.
            inputs = tensors[:-1]
            channels = 2 * m if op == "dinput" else rank * m
            if (
                choice
                and tensors[0].shape[-1] == channels
                and all(t.dtype == DTYPE and t.data_ptr() % 16 == 0 for t in inputs)
                and (op != "dinput" or tensors[-1].shape[-1] == rank * m)
            ):
                candidate, kernel, smem = prepare(shape, op, choice)
                if choice in WEIGHT_STREAM:
                    ranks, _, warps, vec, _ = WEIGHT_STREAM[choice]
                    grid = DIV(m, 32 * vec) * DIV(rank, ranks), grid[1], 1
                elif op == "dweight":
                    warps = TILES[choice].warps
                    grid = DIV(m, 32), grid[1], 1
                else:
                    tile = TILES[choice]
                    warps = tile.warps
                    cols = tile.columns * warps if tile in RING else tile.columns
                    grid = DIV(m, 32), DIV(h, tile.rows) * DIV(w, cols), b
                counts[f"{key}/{op}/{choice}"] += 1
                return Launch(candidate, kernel, grid, tensors, sizes, warps * 32, smem)()
        return original(module, op, grid, tensors, sizes, stream=stream, threads=threads)

    reader._launch_reader_cuda = route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "fold", "dynamo", "select"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.action == "select":
        write_json(args.out, select_policy(args.policy))
        return
    counts: Counter = Counter()
    if args.policy:
        install(json.loads(args.policy.read_text(encoding="utf-8")), counts)
    options = argparse.Namespace(
        batch=args.batch,
        variant=None if args.action == "dynamo" else VARIANT,
        profile=True,
        out=args.out,
    )
    action = {"train": bench.train_arm, "fold": bench.infer_fold, "dynamo": bench.dynamo_check}
    result = action[args.action](options)
    result["prototype_dispatches"] = dict(counts)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
