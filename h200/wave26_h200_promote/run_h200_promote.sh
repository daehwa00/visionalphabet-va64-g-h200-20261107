#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
runner="$PWD/h200/wave26_h200_promote"
out=$(mktemp -d /app/output/va-t93-h200.XXXXXX)
data="$out/results"
mkdir "$data"
export LC_ALL=C PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$out/inductor" VA_T93_BENCH="$out/bench_h200.py"
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED LNET_READER_VEC_MODE
export LNET_READER_VEC=auto

finish() {
    local rc=$? archive_rc=0 part
    trap - EXIT
    set +e
    printf '%d\n' "$rc" > "$data/exit-code.txt"
    printf '%d\n' "$SECONDS" > "$data/elapsed-seconds.txt"
    # The provisioning bot posts only the final 65,000 characters.
    # Keep measured JSON (including full kernels for comparison arms) and retain
    # bounded diagnostics. Full process logs remain in the job output directory.
    transport="$out/transport"
    mkdir "$transport"
    cp "$data"/*.json "$data"/*.tsv "$data"/*.txt "$transport/" 2>/dev/null
    for log in "$data"/*.log; do
        if [[ -s $log ]]; then
            tail -c 3500 "$log" > "$transport/${log##*/}"
        fi
    done
    tar -C "$transport" -cf - . | xz -9e > "$out/task93.tar.xz"
    archive_rc=$?
    if (( archive_rc == 0 )); then
        sha256sum "$out/task93.tar.xz" | awk '{print "VA_T93_SHA256 " $1}'
        base64 -w 240 "$out/task93.tar.xz" | split -l 160 -a 4 -d - "$out/chunk-"
        for part in "$out"/chunk-*; do
            printf 'VA_T93_PART %s\n' "${part##*-}"
            sed 's/^/VA_T93_XZ /' "$part"
        done
        printf 'VA_T93_END %s\n' "$(find "$out" -maxdepth 1 -name 'chunk-*' | wc -l)"
    fi
    printf 'Task 93 result %s, runner exit %d\n' "$out" "$rc"
    cat "$data/gates.json" "$data/processes.tsv" 2>/dev/null
    if (( rc == 0 && archive_rc != 0 )); then exit "$archive_rc"; fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM

variant=17-p1-nr4-1024d-v3
ref=5ec4d31eb4c8e3075af24739db9d10ace2fd1c27
baseline=039440f2c0a9a9d006948fcbbf45dd5c96e36efb
mkdir "$out/wave21c_base"
git archive "$baseline" src scripts | tar -x -C "$out/wave21c_base"
git show "$ref:h200/liverms_v2_opt/bench_h200.py" > "$VA_T93_BENCH"
printf 'benchmark=%s\nbaseline=%s\ncandidate=%s\n' "$ref" "$baseline" "$(git rev-parse HEAD)" > "$data/commits.txt"
ready() {
    test -x "$1" && "$1" -c '
import sys
import numpy, pytest, timm, torch, torchvision, triton
raise SystemExit(sys.version_info < (3, 13) or torch.__version__ != "2.9.1+cu128"
                 or triton.__version__ != "3.5.1" or not torch.cuda.is_available())
' 2>/dev/null
}

unset PYTHON_BIN
for candidate in \
    /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python \
    /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python \
    /app/output/va-t85-h200.*/environment/venv/bin/python \
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
    timeout --kill-after=5s 45s python3 -m pip install -q --disable-pip-version-check --target "$setup/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$setup/bootstrap" python3 -m uv "$@"; }
    timeout --kill-after=5s 60s env PYTHONPATH="$setup/bootstrap" python3 -m uv python install 3.13.11
    uv_run venv --python 3.13.11 "$setup/venv"
    export PYTHON_BIN="$setup/venv/bin/python"
    timeout --kill-after=5s 240s env PYTHONPATH="$setup/bootstrap" python3 -m uv pip install --python "$PYTHON_BIN" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$PYTHON_BIN"
fi > "$data/environment.log" 2>&1

export PYTHONPATH="$PWD/src:$PWD/scripts"
export LNET_CAPTURE_SAFE_ORTHOGONAL=off

run_capped() {
    local limit=$1 name=$2 rc=0
    shift 2
    timeout --kill-after=10s "${limit}s" "$@" > "$data/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\n' "$name" "$rc" "$SECONDS" >> "$data/processes.tsv"
}

config() {
    local arm=$1 root=$PWD
    export LNET_SCAN_W25A=off LNET_READER_W25B=off LNET_FUSED_STEM_CONV=off
    export LNET_VSCAN_MMA=off LNET_CAPTURE_SAFE_ORTHOGONAL=off LNET_READER_VEC_MODE=auto
    if [[ $arm == wave21c_base ]]; then
        root="$out/wave21c_base"
    else
        export LNET_READER_W25B=auto LNET_FUSED_STEM_CONV=auto
        if [[ $arm != wave25bc ]]; then export LNET_SCAN_W25A=auto; fi
        if [[ $arm == wave25bc_hscan8_ps14 ]]; then export LNET_CAPTURE_SAFE_ORTHOGONAL=on; fi
    fi
    export PYTHONPATH="$root/src:$root/scripts"
}

bench() {
    local limit=$1 name=$2 command=$3
    shift 3
    run_capped "$limit" "$name" "$PYTHON_BIN" "$runner/audit.py" "$command" \
        --out "$data/$name.json" "$@"
}

train() (
    local batch=$1 arm=$2 name=$3
    config "$arm"
    opts=(--profile)
    if [[ $name != *_sweep || $batch == 128 ]]; then opts+=(--full-kernels); fi
    bench 360 "b${batch}_$name" train-arm --variant "$variant" --batch "$batch" "${opts[@]}"
)

bench 120 device probe
"$PYTHON_BIN" "$runner/results.py" device "$data"
# Preserve the original failure's traceback independently of promotion tests.
mkdir "$out/original-unroll"
git archive 72f957c9 src scripts tests | tar -x -C "$out/original-unroll"
(
    cd "$out/original-unroll"
    run_capped 120 original-unroll env PYTHONPATH=src:scripts LNET_SCAN_W25A=off \
        "$PYTHON_BIN" -m pytest -q tests/test_general_cuda_scans.py -k interleaved-1
)
for arm in wave21c_base wave25bc wave25bc_hscan8 wave25bc_hscan8_ps14; do
    train 1024 "$arm" "$arm"
done

best=$("$PYTHON_BIN" "$runner/results.py" select "$data")
if [[ -z $best ]]; then
    printf '%s\n' 'No eligible candidate' > "$data/skip.txt"
    exit 1
fi
train 1024 "$best" best_rev
train 1024 wave21c_base wave21c_base_rev

(
    config "$best"
    bench 120 best-probe probe
    bench 360 dynamo dynamo
    bench 360 fold infer-fold --variant "$variant" --batch 2
    tests_start=$SECONDS
    # Unit tests of the explicit preparation API manage their own orthogonal policy.
    run_capped 600 cuda-tests env -u LNET_CAPTURE_SAFE_ORTHOGONAL "$PYTHON_BIN" -m pytest -v -ra \
        tests/test_general_cuda_scans.py tests/test_factorized_reader_kernels.py \
        tests/test_fused_conv_stem_ln_gelu.py tests/test_capture_safe_orthogonal.py \
        tests/test_p1_nr4.py tests/test_live_complex_rms.py
    pytest_rc=$(awk -F '\t' '$1 == "cuda-tests" {print $2}' "$data/processes.tsv")
    bench 30 pytest wrap-pytest --log "$data/cuda-tests.log" --exit-code "$pytest_rc" \
        --elapsed "$((SECONDS - tests_start))"
)
for batch in 128 512 1024 2048; do
    for arm in wave21c_base "$best"; do
        train "$batch" "$arm" "${arm}_sweep"
    done
done
"$PYTHON_BIN" "$runner/results.py" regressions "$data"
