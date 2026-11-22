#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src:$PWD/scripts"
export WANDB_MODE=disabled OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
out=$(mktemp -d /app/output/va-t51-wave13.XXXXXX)
echo "output=$out"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/va-t51-wave13.XXXXXX")
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
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5c11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$run_python"
fi
"$run_python" h200/wave13_vscan_probe/bench.py --out "$out/isolated.jsonl" | tee "$out/isolated.log"
"$run_python" -c 'import json,sys; print(json.dumps([json.loads(r) for r in sys.stdin]))' \
    < "$out/isolated.jsonl" > "$out/isolated.json"
bench=h200/liverms_v2_opt/bench_h200.py
mkdir -p "$out/arms" "$out/infer"
for policy in 0 1; do
    export LNET_VSCAN_DECOUPLED=$policy
    for batch in 1024 128 256 512; do
        "$run_python" "$bench" train-arm --out "$out/arms/p1_b${batch}_${policy}.json" \
            --variant 17-p1-nr4-1024d-v3 --batch "$batch" --profile
    done
    for variant in 17-live-complex-rms-s03 17-e2-multiscale-s3s4-live-rms-v2; do
        "$run_python" "$bench" train-arm --out "$out/arms/${variant}_b1024_${policy}.json" \
            --variant "$variant" --batch 1024
    done
    "$run_python" "$bench" infer-fold --out "$out/infer/p1_b1024_${policy}.json" \
        --variant 17-p1-nr4-1024d-v3 --batch 1024
done
"$run_python" "$bench" dynamo --out "$out/dynamo.json"
