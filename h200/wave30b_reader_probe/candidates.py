"""Assemble candidate reader tiles on SM 9.0 without a CUDA device."""

# ruff: noqa: ANN001, ANN201, S603, SLF001, T201
import argparse
import itertools
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from offline import OfflineModule
from replay import resources

from lnet import factorized_reader_kernels as reader

PTXAS = "/usr/local/cuda-12.8/bin/ptxas"


ROLLING_LOOP = """            for (int x = 0; x < width; ++x) {
                Window right[Rows + 2][2] = {};
                auto load_right = [&](int i) {
                    const int gy = y + i - 1;
                    if (gy >= 0 && gy < height && x + 1 < width) {
                        const G* g = grad + (base + gy * width + x + 1) * 2 * modes + mode;
                        right[i][0] = window_load<Vec, ReadOnly, ConvertWindow>(g);
                        right[i][1] = window_load<Vec, ReadOnly, ConvertWindow>(g + modes);
                    }
                };
                #pragma unroll
                for (int i = 0; i < 2; ++i) load_right(i);
                #pragma unroll
                for (int j = 0; j < Rows; ++j) {
                    load_right(j + 2);
                    if (y + j < height) {
                        V v[RankTile];
                        #pragma unroll
                        for (int r = 0; r < RankTile; ++r) {
                            const T* current = p + ((j * width + x) * R + r) * modes;
                            v[r] = r0 + r < R ? mode_load<Vec, T, ReadOnly>(current) : V{};
                        }
                        #pragma unroll
                        for (int ky = 0; ky < 3; ++ky) {
                            const int i = j + 2 - ky;
                            const V l[] = {window_values<Vec, G>(left[i][0]),
                                window_values<Vec, G>(left[i][1])};
                            const V m[] = {window_values<Vec, G>(middle[i][0]),
                                window_values<Vec, G>(middle[i][1])};
                            const V g[] = {window_values<Vec, G>(right[i][0]),
                                window_values<Vec, G>(right[i][1])};
                            #pragma unroll
                            for (int r = 0; r < RankTile; ++r) {
                                sums_r[ky][0][r] = fmaf(v[r], g[0], sums_r[ky][0][r]);
                                sums_r[ky][1][r] = fmaf(v[r], m[0], sums_r[ky][1][r]);
                                sums_r[ky][2][r] = fmaf(v[r], l[0], sums_r[ky][2][r]);
                                sums_i[ky][0][r] = fmaf(v[r], g[1], sums_i[ky][0][r]);
                                sums_i[ky][1][r] = fmaf(v[r], m[1], sums_i[ky][1][r]);
                                sums_i[ky][2][r] = fmaf(v[r], l[1], sums_i[ky][2][r]);
                            }
                        }
                    }
                    // Row j has served its last tap in this column.
                    left[j][0] = middle[j][0];
                    left[j][1] = middle[j][1];
                    middle[j][0] = right[j][0];
                    middle[j][1] = right[j][1];
                }
                #pragma unroll
                for (int i = Rows; i < Rows + 2; ++i) {
                    left[i][0] = middle[i][0];
                    left[i][1] = middle[i][1];
                    middle[i][0] = right[i][0];
                    middle[i][1] = right[i][1];
                }
            }
        }
    }
"""


def compile_candidate(job):
    op, height, dtype, vec, rows, columns, warps, rank_tile, filters = job
    modes = 128 if height == 7 else 160
    weight = op == "dweight"
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.object(reader, "_CudaModule", OfflineModule),
        TemporaryDirectory() as scratch,
    ):
        path = Path(scratch)
        module = reader._reader_cuda_module(
            0, (1024, height, height, modes), columns, 4, 3, (dtype,) * 3,
            strict=False, spatial_geometry=(rows, warps) if not weight else (2, 1),
            weight_geometry=(rank_tile, rows, warps) if weight else None,
            spatial_vec=1 if weight else vec, weight_vec=vec if weight else 1,
            register_filters=filters, keep_real=0, updated_spatial=True,
            weight_read_only=vec != 2,
        )
        reader._reader_cuda_module.cache_clear()
        ptx = path / "reader.ptx"
        ptx.write_bytes(module.image)
        log = subprocess.run(
            [PTXAS, "-v", "-arch=sm_90", str(ptx), "-o", str(path / "reader.cubin")],
            check=True, capture_output=True, text=True,
        )
        return [*job, resources(log.stderr)[op]]


def jobs():  # noqa: C901
    for dtype in ("__nv_bfloat16", "__half"):
        for height, rows in ((56, 8), (28, 7)):
            for columns in (height // 2, height):
                for warps in (1, 2, 4):
                    yield "forward", height, dtype, 2, rows, columns, warps, 4, 0
        for height in (56, 28, 14, 7):
            for rows, columns, warps, filters in itertools.product(
                (4, 7), (28, 56) if height == 56 else (height,), (1, 2, 4), (0, 1),
            ):
                yield "forward", height, dtype, 1, rows, columns, warps, 4, filters
            for warps in (1, 2, 4):
                for columns in (height,) if height < 56 else (28, 56):
                    yield "dinput", height, dtype, 1, 4 if height == 28 else 7, columns, warps, 4, 0
            for rows in (2, 4, 7, 8):
                if rows <= height:
                    yield "dweight", height, dtype, 1, rows, 7, 4, 4, 0
                    for warps in (2, 4):
                        yield "dweight", height, dtype, 2, rows, 14, warps, 2, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling", action="store_true")
    args = parser.parse_args()
    selected = jobs()
    if args.rolling:
        source = reader._READER_CUDA
        start = source.index("            for (int x = 0; x < width; ++x)", source.index(
            "void real_weight_gradient_3x3",
        ))
        end = source.index("    // Stage a subset", start)
        reader._READER_CUDA = source[:start] + ROLLING_LOOP + source[end:]
        selected = (job for job in selected if job[0] == "dweight" and job[3] == 2)
    with ProcessPoolExecutor(max_workers=12, mp_context=get_context("fork")) as pool:
        for result in pool.map(compile_candidate, selected):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
