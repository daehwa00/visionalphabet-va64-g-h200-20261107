"""Audit the existing stem NVRTC geometry on candidate device targets."""

# ruff: noqa: S603, T201
import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from lnet.fused_stem_ops import _compile_ptx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxas", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="wave24c-ptx-") as tmp:
        source = Path(tmp) / "stem.ptx"
        binary = Path(tmp) / "stem.cubin"
        for arch in ((8, 6), (8, 9), (9, 0)):
            for dtype in ("__nv_bfloat16", "__half"):
                for width in (32, 64):
                    code = _compile_ptx(
                        arch,
                        width,
                        (dtype, "float", "float", dtype),
                        vector=True,
                        conv_bias_type="float",
                        round_affine=False,
                    )
                    source.write_bytes(code.value)
                    print(arch, dtype, width, flush=True)
                    subprocess.run(
                        [args.ptxas, "-v", f"-arch=sm_{arch[0]}{arch[1]}", source, "-o", binary],
                        check=True,
                    )


if __name__ == "__main__":
    main()
