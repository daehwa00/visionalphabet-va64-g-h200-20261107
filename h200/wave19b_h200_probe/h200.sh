#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

mkdir -p /app/output
out=$(mktemp -d /app/output/va-t72-wave19b-h200.XXXXXX)
export LC_ALL=C
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED LNET_READER_VEC
export TORCHINDUCTOR_CACHE_DIR="$out/inductor"

baseline=42ff1ee5170892b94d44a4a17106af28b71400ab
variant=17-p1-nr4-1024d-v3
bench=h200/liverms_v2_opt/bench_h200.py
wave17b=fda2db59b0e40e76adacb8b81c5a53143dee0fb1
probe=h200/wave19b_h200_probe/results.py
CAND_PP="$PWD/src:$PWD/scripts"

finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf '%d\n' "$rc" > "$out/exit-code.txt"
    printf 'Wave 19B results %s, runner exit %d\n' "$out" "$rc"
    python3 "$probe" summary "$out"
    (
        cd -- "$out" || exit
        shopt -s nullglob
        tar -cJf - -- ./*.json ./*.jsonl ./*.log ./*.txt
    ) | base64 -w 120 | sed 's/^/VA_T72_XZ /'
    local archive_rc=$?
    if (( rc == 0 && archive_rc != 0 )); then
        exit "$archive_rc"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM
printf 'not-run\n' > "$out/pytest_rc.txt"
printf 'head=%s\nwave14a=%s\nwave17b=%s\n' "$(git rev-parse HEAD)" "$baseline" "$wave17b" > "$out/commits.txt"
mkdir "$out/wave14a" "$out/wave17b"
git archive "$baseline" src scripts | tar -x -C "$out/wave14a"
git archive "$wave17b" src scripts | tar -x -C "$out/wave17b"

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

run_candidate() {
    LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$@"
}

train() {
    local batch=$1 arm=$2 name=$3 pp vec
    shift 3
    case "$arm" in
        baseline_w14a) pp="$out/wave14a/src:$out/wave14a/scripts"; vec=1 ;;
        wave17b) pp="$out/wave17b/src:$out/wave17b/scripts"; vec=auto ;;
        wave19b) pp=$CAND_PP; vec=auto ;;
    esac
    LNET_READER_VEC=$vec PYTHONPATH="$pp" "$PYTHON_BIN" "$bench" train-arm \
        --variant "$variant" --batch "$batch" --out "$out/$name.json" "$@"
}

echo "[1] Device probe"
run_candidate "$bench" probe --out "$out/device.json"
python3 "$probe" device "$out"

echo "[2] B=1024 forward and reversed A/B"
for arm in baseline_w14a wave17b wave19b; do
    train 1024 "$arm" "b1024_$arm" --profile
done
for arm in wave19b wave17b; do
    train 1024 "$arm" "b1024_${arm}_rev" --profile
done

echo "[3] Batch sweep"
for b in 128 256 512; do
    for arm in baseline_w14a wave17b wave19b; do
        train "$b" "$arm" "b${b}_$arm"
    done
done

echo "[4] CUDA Graph reader BF16 and FP16"
for shape in '56 56 160 4' '28 28 320 4' '14 14 640 4' '7 7 640 4'; do
    read -r -a dims <<< "$shape"
    run_candidate h200/wave17b_vec_reader_probe/bench.py bench \
        --batch 1024 --samples 10 --shape "${dims[@]}" --arms auto 1 2 \
        --baseline "$out/wave14a/src/lnet/factorized_reader_kernels.py" \
        --out "$out/reader_micro_${shape// /x}.jsonl"
done

echo "[5] Dynamo and fold correctness"
run_candidate "$bench" dynamo --out "$out/dynamo.json"
run_candidate "$bench" infer-fold --variant "$variant" --batch 2 --out "$out/fold.json"

echo "[6] CUDA pytest, 13m cap"
pytest_rc=0
LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" timeout --kill-after=10s 780s \
    "$PYTHON_BIN" -m pytest -q tests/test_factorized_reader_kernels.py tests/test_p1_nr4.py \
    tests/test_general_cuda_scans.py > "$out/pytest.log" 2>&1 || pytest_rc=$?
printf '%d\n' "$pytest_rc" > "$out/pytest_rc.txt"
tail -n 20 "$out/pytest.log"
python3 "$probe" gates "$out"
