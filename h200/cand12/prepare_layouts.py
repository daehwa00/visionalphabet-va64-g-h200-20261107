"""Materialize independent candidates from the pinned cand12 snapshot."""
from pathlib import Path
import shutil
import subprocess
import sys


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


root = Path(sys.argv[1]).resolve()
for name, modes, warps, lines in (("transfer", 4, 4, 4), ("h8", 8, 4, 4), ("h8w8", 8, 8, 8)):
    tree = root / name
    shutil.copytree("code-cand12", tree)
    with open("layout-4090.patch") as patch:
        subprocess.run(["patch","-s","-p1"], stdin=patch, cwd=tree, check=True)
    path = tree / "src/lnet/product_scan_ops.py"
    src = replace_once(path.read_text(), "(128, 56, 56, 160)", "(1024, 56, 56, 160)")
    src = replace_once(src, 'torch.cuda.get_device_name(source_real_a.device) == "NVIDIA GeForce RTX 4090"', '"H200" in torch.cuda.get_device_name(source_real_a.device)')
    src = replace_once(src, "lines_per_group = 4", f"lines_per_group = {lines}")
    src = replace_once(src, '_RTX4090_PATH_BACKWARD_GEOMETRY = LaunchGeometry.build(\n    num_warps=4, blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 4},', f'_H200_TRIAL = LaunchGeometry.build(\n    num_warps={warps}, blocks={{"BLOCK_LINES": 1, "BLOCK_MODES": {modes}}},')
    path.write_text(src.replace("(_RTX4090_PATH_BACKWARD_GEOMETRY,)", "(_H200_TRIAL,)"))
