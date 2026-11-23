"""Assemble the weight-window ablations for the three CUDA targets."""

# ruff: noqa: S603, SLF001
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from offline import OfflineModule
from replay import SM, resources
from screen import REAL
from weight_window import compile_weight

from lnet import factorized_reader_kernels as reader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--sm", choices=SM, nargs="+", default=tuple(SM))
    args = parser.parse_args()
    configurations = (
        (4, 4, False, False), (4, 4, True, False), (4, 4, False, True), (7, 4, False, False),
        (8, 4, False, False), (14, 4, False, False), (14, 8, False, False),
    )
    with (
        TemporaryDirectory() as tmp,
        args.out.open("w") as output,
        patch.object(reader, "_CudaModule", OfflineModule),
    ):
        ptx, cubin = Path(tmp) / "reader.ptx", Path(tmp) / "reader.cubin"
        for target in args.sm:
            capability = tuple(map(int, target))
            batch = 1024 if target == "90" else 128
            reader._reader_cuda_module.cache_clear()
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                cases = itertools.product(REAL, ("__nv_bfloat16", "__half"), configurations)
                for (height, width, modes, rank), dtype, (rows, warps, readonly, convert) in cases:
                    shape = batch, height, width, modes
                    module = compile_weight(
                        0, shape, reader._reader_columns(width), rank, 3, (dtype,) * 3,
                        strict=False, weight_geometry=(2, rows, warps), weight_vec=2,
                        weight_read_only=readonly, convert=convert,
                    )
                    ptx.write_bytes(module.image)
                    result = subprocess.run(
                        [args.ptxas, "-v", f"-arch=sm_{target}", ptx, "-o", cubin],
                        capture_output=True, text=True, check=True,
                    )
                    record = dict(sm=target, shape=shape, dtype=dtype, rows=rows, warps=warps,
                                  readonly=readonly, convert=convert,
                                  **resources(result.stderr)["dweight"])
                    print(json.dumps(record), file=output, flush=True)


if __name__ == "__main__":
    main()
