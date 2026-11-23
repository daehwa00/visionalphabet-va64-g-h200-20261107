#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

mkdir -p /app/output
out=$(mktemp -d /app/output/va-t85-h200.XXXXXX)
export LC_ALL=C
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED LNET_READER_VEC
export TORCHINDUCTOR_CACHE_DIR="$out/inductor"

variant=17-p1-nr4-1024d-v3
bench="$PWD/h200/liverms_v2_opt/bench_h200.py"
results="$PWD/h200/wave21_h200_bundle/results.py"
wave17b=fda2db59b0e40e76adacb8b81c5a53143dee0fb1
wave20b=ad6fb7835a7a0272b2a70144ee2f70abd0763499
wave21c=039440f2c0a9a9d006948fcbbf45dd5c96e36efb
combined=$(git rev-parse HEAD)

finish() {
    local rc=$? archive_rc
    trap - EXIT
    set +e
    printf '%d\n' "$rc" > "$out/exit-code.txt"
    printf 'Task 85 result %s, runner exit %d\n' "$out" "$rc"
    (
        cd -- "$out" || exit
        shopt -s nullglob
        tar -cJf - -- ./*.json ./*.txt
    ) | base64 -w 240 | sed 's/^/VA_T85_XZ /'
    archive_rc=$?
    if (( rc == 0 && archive_rc != 0 )); then
        exit "$archive_rc"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM
printf 'wave17b=%s\nwave20b=%s\nwave21c=%s\ncombined=%s\n' \
    "$wave17b" "$wave20b" "$wave21c" "$combined" > "$out/commits.txt"

for arm in wave17b wave20b wave21c; do
    mkdir "$out/$arm"
    git archive "${!arm}" src scripts | tar -x -C "$out/$arm"
done

ready() {
    test -x "$1" && "$1" -c '
import sys
import numpy, pytest, timm, torch, torchvision, triton
raise SystemExit(sys.version_info < (3, 13) or torch.__version__ != "2.9.1+cu128"
                 or triton.__version__ != "3.5.1" or not torch.cuda.is_available())
' 2>/dev/null
}

for candidate in \
    /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python \
    /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python \
    "$(command -v python3)"; do
    if ready "$candidate"; then
        export PYTHON_BIN=$candidate
        break
    fi
done
if [[ -z ${PYTHON_BIN:-} ]]; then
    setup="$out/environment"
    mkdir -p "$setup"
    export UV_PYTHON_INSTALL_DIR="$setup/python" UV_CACHE_DIR="$setup/uv-cache"
    python3 -m pip install -q --disable-pip-version-check --target "$setup/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$setup/bootstrap" python3 -m uv "$@"; }
    uv_run python install 3.13.11
    uv_run venv --python 3.13.11 "$setup/venv"
    export PYTHON_BIN="$setup/venv/bin/python"
    uv_run pip install --python "$PYTHON_BIN" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$PYTHON_BIN"
fi

export LNET_READER_VEC=auto
export PYTHONPATH="$PWD/src:$PWD/scripts"

train() {
    local batch=$1 arm=$2 name=$3 paths=$PYTHONPATH
    shift 3
    if [[ $arm != combined ]]; then
        paths="$out/$arm/src:$out/$arm/scripts"
    fi
    printf 'B=%s %s\n' "$batch" "$name"
    PYTHONPATH="$paths" "$PYTHON_BIN" "$bench" train-arm \
        --variant "$variant" --batch "$batch" --out "$out/b${batch}_$name.json" "$@"
}

"$PYTHON_BIN" "$bench" probe --out "$out/device.json"
"$PYTHON_BIN" "$results" device "$out"

for arm in wave17b wave20b wave21c combined; do
    train 1024 "$arm" "$arm" --profile --full-kernels
done
for arm in combined wave17b; do
    train 1024 "$arm" "${arm}_rev"
done
for batch in 128 256 512 2048; do
    for arm in wave17b combined; do
        train "$batch" "$arm" "$arm"
    done
done

"$PYTHON_BIN" "$bench" dynamo --out "$out/dynamo.json"
"$PYTHON_BIN" "$bench" infer-fold --variant "$variant" --batch 2 --out "$out/fold.json"

pytest_rc=0
SECONDS=0
"$PYTHON_BIN" -m pytest -v -ra \
    tests/test_factorized_reader_kernels.py tests/test_general_cuda_scans.py \
    tests/test_p1_nr4.py tests/test_live_complex_rms.py > "$out/pytest.log" 2>&1 || pytest_rc=$?
tail -n 16 "$out/pytest.log" > "$out/pytest-tail.txt"
"$PYTHON_BIN" "$bench" wrap-pytest --log "$out/pytest.log" --exit-code "$pytest_rc" \
    --elapsed "$SECONDS" --out "$out/pytest.json"
"$PYTHON_BIN" "$results" gates "$out"
