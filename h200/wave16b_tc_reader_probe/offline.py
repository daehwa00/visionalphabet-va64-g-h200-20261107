"""Compile the production MMA source with nvcc/ptxas, without a CUDA device."""

# ruff: noqa: S603, S607
import argparse
import ast
import re
import subprocess
from pathlib import Path


def compile_source(out: Path) -> None:
    tree = ast.parse(Path("src/lnet/factorized_reader_kernels.py").read_text())
    names = ("_READER_CUDA_TYPES", "_READER_MMA_CUDA", "_READER_REDUCE_CUDA")
    templates = {
        item.targets[0].id: ast.literal_eval(item.value)
        for item in tree.body
        if isinstance(item, ast.Assign)
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id in names
    }
    out.mkdir(parents=True, exist_ok=True)
    rows = ["sm,dtype,rank,shape,kernel,registers,smem_bytes,spill_store,spill_load,hmma"]
    for dtype, cuda_type in (("bf16", "__nv_bfloat16"), ("fp16", "__half")):
        for rank in (2, 4, 8):
            shapes = [(0, 0, 0, 0)]
            if rank == 4:
                shapes += [(128, 56, 56, 160), (1024, 56, 56, 160), (4, 512, 768, 48)]
            for shape in shapes:
                macros = {
                    "BFLOAT16": int(dtype == "bf16"),
                    "RANK": rank,
                    "KERNEL": 3,
                    "MMA_MODES": 16,
                    "MMA_ROWS": 4,
                    "MMA_PIXELS": 16,
                    "MMA_SHIFTS": 4,
                    **dict(zip(("BATCH", "HEIGHT", "WIDTH", "MODES"), shape, strict=True)),
                    "INPUT": cuda_type,
                    "WEIGHT": cuda_type,
                    "GRAD": cuda_type,
                }
                label = "dynamic" if not shape[0] else "x".join(map(str, shape))
                source = out / f"{dtype}-r{rank}-{label}.cu"
                source.write_text(
                    "".join(f"#define READER_{key} {value}\n" for key, value in macros.items())
                    + "".join(templates[name] for name in names),
                )
                for sm in (86, 89, 90):
                    prefix = out / f"sm{sm}-{dtype}-r{rank}-{label}"
                    ptx, cubin = prefix.with_suffix(".ptx"), prefix.with_suffix(".cubin")
                    subprocess.run(
                        [
                            "nvcc",
                            "-std=c++17",
                            "-ptx",
                            f"-arch=sm_{sm}",
                            str(source),
                            "-o",
                            str(ptx),
                        ],
                        check=True,
                    )
                    build = subprocess.run(
                        [
                            "ptxas",
                            "-v",
                            f"-arch=sm_{sm}",
                            str(ptx),
                            "-o",
                            str(cubin),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    prefix.with_suffix(".ptxas.log").write_text(build.stderr)
                    build.check_returncode()
                    sass = subprocess.check_output(
                        ["cuobjdump", "--dump-sass", str(cubin)], text=True
                    )
                    prefix.with_suffix(".sass").write_text(sass)
                    for kernel in ("forward", "shift_forward", "dinput", "dweight"):
                        name = f"reader_mma_{kernel}"
                        info = build.stderr.split(f"Function properties for {name}\n")[1].split(
                            "Compiling entry function",
                        )[0]
                        store, load = re.search(
                            r"(\d+) bytes spill stores, (\d+) bytes spill loads",
                            info,
                        ).groups()
                        registers = re.search(r"Used (\d+) registers", info)[1]
                        smem = re.search(r"(\d+) bytes smem", info)
                        instructions = sass.split(f"Function : {name}\n")[1].split("Function :")[0]
                        hmma = instructions.count("HMMA.16816")
                        shared = smem[1] if smem else 0
                        rows.append(
                            f"{sm},{dtype},{rank},{label},{kernel},{registers},"
                            f"{shared},{store},{load},{hmma}",
                        )
    (out / "resources.csv").write_text("\n".join(rows) + "\n")
    if any(row.split(",")[-3:-1] != ["0", "0"] or not int(row.split(",")[-1]) for row in rows[1:]):
        raise ValueError("Expected HMMA instructions and no spills in each MMA kernel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    compile_source(parser.parse_args().out)
