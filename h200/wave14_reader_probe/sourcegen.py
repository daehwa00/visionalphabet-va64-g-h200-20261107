"""Define Wave 14B variants and assemble the same source for NVRTC and ptxas."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SHAPES = (
    (1024, 56, 56, 160, 4),
    (1024, 28, 28, 160, 4),
    (1024, 14, 14, 160, 4),
    (1024, 7, 7, 128, 4),
    (4, 128, 192, 160, 4),
    (4, 512, 768, 48, 4),
    (4, 96, 128, 80, 2),
)


@dataclass(frozen=True)
class Tile:
    name: str
    rows: int
    columns: int
    warps: int
    vec: int = 1
    async_copy: bool = True
    row_block: int = 1
    reuse: bool = False

    def spatial_smem(self, rank: int, *, backward: bool) -> int:
        return (self.rows + 2) * (self.columns + 2) * (
            2 if backward else rank
        ) * 64 + 2 * rank * 9 * 128

    def weight_smem(self, rank: int) -> int:
        groups = self.warps * self.vec // rank
        projected = self.rows * self.columns * rank * 64
        reduction = groups * rank * 18 * 128
        return max(projected, reduction) + (self.rows + 2) * (self.columns + 2) * 128


RING = (
    Tile("ring4x7", 4, 7, 4),
    Tile("ring4x14", 4, 14, 4),
    Tile("ring4x28", 4, 28, 2),
    Tile("ring4x56", 4, 56, 1),
    Tile("ring8x14", 8, 14, 4),
)
SPATIAL = (
    Tile("smem8x14_scalar", 8, 14, 4, row_block=2),
    Tile("smem8x14_pair", 8, 14, 4, 2),
    Tile("smem8x14_ldg128", 8, 14, 4, 2, async_copy=False),
    Tile("smem8x16_pair", 8, 16, 4, 2),
    Tile("smem8x28_pair", 8, 28, 4, 2),
    Tile("smem4x28_pair", 4, 28, 4, 2),
)
WEIGHT = (
    Tile("weight8x14_pair", 8, 14, 4, 2),
    Tile("weight8x14_ldg128", 8, 14, 4, 2, async_copy=False),
    Tile("weight8x28_pair", 8, 28, 4, 2),
    Tile("weight14x14_pair", 14, 14, 4, 2),
    Tile("weight8x16_scalar", 8, 16, 8),
    Tile("weight8x14_reuse", 8, 14, 4, 2, reuse=True),
    Tile("weight8x28_reuse", 8, 28, 4, 2, reuse=True),
)
# Candidate names map to (rank tile, rows per warp, warps per CTA, vectors per lane, worker waves).
WEIGHT_STREAM = {
    "weight_t1r4_vec2": (1, 4, 4, 2, 10),
    "weight_t1r8_vec2": (1, 8, 4, 2, 10),
    "weight_t1r14_vec2": (1, 14, 4, 2, 10),
    "weight_t2r8_scalar": (2, 8, 4, 1, 10),
    "weight_t2r14_scalar": (2, 14, 4, 1, 10),
}


def baseline_sources() -> tuple[str, str]:
    tree = ast.parse((ROOT / "src/lnet/factorized_reader_kernels.py").read_text(encoding="utf-8"))
    values = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            if target in {"_READER_CUDA_TYPES", "_READER_CUDA"}:
                values[target] = ast.literal_eval(statement.value)
    return values["_READER_CUDA_TYPES"], values["_READER_CUDA"]


def generate(shape: tuple[int, ...], names: set[str], *, native: bool = False) -> str:
    batch, height, width, modes, rank = shape
    # Leave edge width/height static even when there is a partially populated mode warp.
    macros = {
        "BATCH": batch,
        "HEIGHT": height,
        "WIDTH": width,
        "MODES": modes,
        "COLUMNS": 7 if width % 7 == 0 else 8,
        "RANK": rank,
        "KERNEL": 3,
        "STRICT": 0,
        "SPATIAL_ROWS": 4,
        "SPATIAL_WARPS": 4,
        "REGISTER_FILTERS": 0,
        "KEEP_REAL": 0,
        "BFLOAT16": 1,
        "WEIGHT_TILE": 2,
        "WEIGHT_ROWS": 4,
        "WEIGHT_WARPS": 8,
        "WEIGHT_VEC": 1,
        "INPUT": "__nv_bfloat16",
        "WEIGHT": "__nv_bfloat16",
        "GRAD": "__nv_bfloat16",
    }
    definitions, baseline = baseline_sources()
    if not native:
        baseline = baseline[: baseline.index('extern "C" __global__ void reader_forward(')]
    wrappers = []
    for tile in (*RING, *SPATIAL):
        if tile.name not in names:
            continue
        for suffix, backward in (("forward", "false"), ("dinput", "true")):
            if tile in RING:
                launch = f"ring_spatial<{tile.rows},{tile.columns},{tile.warps},{backward}>"
            else:
                async_copy = str(tile.async_copy).lower()
                launch = (
                    f"tiled_spatial<{backward},{tile.rows},{tile.columns},"
                    f"{tile.vec},{async_copy},{tile.row_block}>"
                )
            wrappers.append(f"""
extern "C" __global__ void reader_{tile.name}_{suffix}(
    const Input* p, const Weight* r, const Weight* i, Input* out, int h, int w, int m, int rank
) {{ {launch}(p, r, i, out, h, w, m); }}
""")
    for tile in WEIGHT:
        if tile.name not in names:
            continue
        async_copy = str(tile.async_copy).lower()
        template = (
            f"{tile.rows},{tile.columns},{tile.warps},{tile.vec},{async_copy},"
            f"{str(tile.reuse).lower()}"
        )
        wrappers.append(f"""
extern "C" __global__ void reader_{tile.name}_dweight(
    const Input* p, const Grad* g, float* dw, int h, int w, int m, int b, int rank
) {{ tiled_dweight<{template}>(p, g, dw, h, w, m, b); }}
""")
    header = "".join(f"#define READER_{key} {value}\n" for key, value in macros.items())
    src = "\n".join((HERE / name).read_text(encoding="utf-8") for name in ("ring.cuh", "tiled.cuh"))
    return header + definitions + baseline + src + "".join(wrappers)
