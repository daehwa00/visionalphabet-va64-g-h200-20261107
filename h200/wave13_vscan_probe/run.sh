#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src:$PWD/scripts"
export WANDB_MODE=disabled OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
out=$(mktemp -d /app/output/va-t55-wave13.XXXXXX)
echo "output=$out"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/va-t55-wave13.XXXXXX")
export TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
export RUN_ROOT=$out PERSIST_ROOT=$out
finish() {
    local rc=$?
    python3 h200/liverms_v2_opt/export_result.py "$rc"
    rm -rf -- "$scratch"
}
trap finish EXIT
ready() {
    [[ -x $1 ]] && "$1" -c 'import sys, torch, torchvision, timm; assert sys.version_info >= (3, 13); assert torch.cuda.is_available()'
}
run_python=
for candidate in /app/output/va-h200-{effective1024-r3-combined,r4-marker-fix}-20261119/environment/bin/python "$(command -v python3)"; do
    if ready "$candidate"; then
        run_python=$candidate
        break
    fi
done
if [[ -z $run_python ]]; then
    export UV_PYTHON_INSTALL_DIR="$scratch/python" UV_CACHE_DIR="$scratch/uv-cache"
    python3 -m pip install -q --target "$scratch/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$scratch/bootstrap" python3 -m uv "$@"; }
    uv_run python install 3.13.11
    uv_run venv --python 3.13.11 "$scratch/environment"
    run_python="$scratch/environment/bin/python"
    uv_run pip install --python "$run_python" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$run_python"
fi
bench=h200/liverms_v2_opt/bench_h200.py
mkdir -p "$out/arms" "$out/infer"
if [[ ${1:-} != --confirm ]]; then
    "$run_python" h200/wave13_vscan_probe/bench.py --out "$out/isolated.jsonl" | tee "$out/isolated.log"
    "$run_python" -c 'import json,sys; print(json.dumps([json.loads(r) for r in sys.stdin]))' \
        < "$out/isolated.jsonl" > "$out/isolated.json"
    "$run_python" h200/wave13_vscan_probe/projection.py --quick \
        --variants vector vector-shared4 --out "$out/projection.jsonl"
    "$run_python" -c 'import json,sys; print(json.dumps([json.loads(r) for r in sys.stdin]))' \
        < "$out/projection.jsonl" > "$out/projection.json"
fi
# Reuse the cuDNN autotuner and loaded CUDA kernels between arms.
"$run_python" - "$out" "${1:-}" <<'PYBENCH'
import gc
import os
import runpy
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
confirm = sys.argv[2] == "--confirm"
bench_script = "h200/liverms_v2_opt/bench_h200.py"
sys.path.insert(0, "h200/wave13_vscan_probe")
from metrics import instrument_training

main = runpy.run_path(bench_script)["main"]
main.__globals__["profile_step"] = instrument_training(main.__globals__["profile_step"])
arms = ("0", "chunk", "saved")
variant = "17-p1-nr4-1024d-v3"

def run(arm, command, out, *options):
    os.environ["LNET_VSCAN_DECOUPLED"] = arm
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    print(f"begin {arm} {command} {out.name}", flush=True)
    sys.argv = [bench_script, command, "--out", str(out), *options]
    main()

for batch in ((1024,) if confirm else (1024, 128, 256, 512)):
    for arm in arms:
        run(
            arm, "train-arm", root / "arms" / f"p1_b{batch}_{arm}.json",
            "--variant", variant, "--batch", str(batch), "--profile",
        )
for arm in (("saved",) if confirm else arms):
    run(
        arm, "infer-fold", root / "infer" / f"p1_b1024_{arm}.json",
        "--variant", variant, "--batch", "1024",
    )
    run(arm, "dynamo", root / f"dynamo_{arm}.json")
PYBENCH
