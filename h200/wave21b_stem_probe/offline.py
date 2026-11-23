"""Assemble stem specializations without a CUDA device."""

# ruff: noqa: S603, S607, T201
import argparse
import re
import subprocess
from pathlib import Path

from lnet.fused_stem_ops import _compile_ptx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="store_true", help="compile with the probe patch")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for capability in ((8, 6), (8, 9), (9, 0)):
        sm = "".join(map(str, capability))
        for dtype in ("__nv_bfloat16", "__half", "float"):
            for channels in (32, 64, 96, 160, 53):
                candidate_supported = args.candidate and channels <= 128 and channels % 8 == 0
                candidates = (False, True) if candidate_supported else (False,)
                for candidate in candidates:
                    stem = args.output / f"{sm}-{dtype}-{channels}-{int(candidate)}"
                    options = {"parallel_reduction": candidate} if args.candidate else {}
                    image = _compile_ptx(
                        capability,
                        channels,
                        (dtype, "float", "float", dtype),
                        vector=channels % 8 == 0,
                        conv_bias_type="float",
                        round_affine=False,
                        **options,
                    )
                    ptx, cubin = stem.with_suffix(".ptx"), stem.with_suffix(".cubin")
                    ptx.write_bytes(image.value)
                    assembled = subprocess.run(
                        ["ptxas", "-v", f"-arch=sm_{sm}", str(ptx), "-o", str(cubin)],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    sass = subprocess.check_output(["nvdisasm", str(cubin)], text=True)
                    widths = sorted(set(re.findall(r"\b(?:LDG|STG)(?:\.[A-Z0-9]+)*", sass)))
                    print(stem.name, " ".join(widths), assembled.stderr, sep="\n", flush=True)


if __name__ == "__main__":
    main()
