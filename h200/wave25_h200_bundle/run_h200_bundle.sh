#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
runner="$PWD/h200/wave25_h200_bundle"
out=$(mktemp -d /app/output/va-t92-h200.XXXXXX)
data="$out/results"
mkdir "$data"
export LC_ALL=C PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$out/inductor" VA_T92_BENCH="$out/bench_h200.py"
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
    tar -C "$data" -cf - . | xz -7e > "$out/task92.tar.xz"
    archive_rc=$?
    if (( archive_rc == 0 )); then
        sha256sum "$out/task92.tar.xz" | awk '{print "VA_T92_SHA256 " $1}'
        base64 -w 240 "$out/task92.tar.xz" | split -l 160 -d - "$out/chunk-"
        for part in "$out"/chunk-*; do
            printf 'VA_T92_PART %s\n' "${part##*-}"
            sed 's/^/VA_T92_XZ /' "$part"
        done
        printf 'VA_T92_END %s\n' "$(find "$out" -maxdepth 1 -name 'chunk-*' | wc -l)"
    fi
    printf 'Task 92 result %s, runner exit %d\n' "$out" "$rc"
    if (( rc == 0 && archive_rc != 0 )); then exit "$archive_rc"; fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM

variant=17-p1-nr4-1024d-v3
ref=5ec4d31eb4c8e3075af24739db9d10ace2fd1c27
declare -A revisions=(
    [wave21c_base]=039440f2c0a9a9d006948fcbbf45dd5c96e36efb
    [wave23a_mma]=e474e4468bdc555ffac1219f8ee308803fedd074
    [wave23b_vec4]=176f9ab834af6ada0fd0fb8d2c960aa94ca1870b
    [wave22cv2_ps14]=83e480f26b7437f8d62a1859234d7725357538b4
)
timeout --kill-after=5s 60s git fetch origin wave21-h200-bundle wave23a-vscan-mma \
    wave23b-reader-vec4 wave22cv2-ps14-cudagraph > "$data/fetch.log" 2>&1
git show "$ref:h200/liverms_v2_opt/bench_h200.py" > "$VA_T92_BENCH"
printf 'benchmark=%s\nbundle=%s\n' "$ref" "$(git rev-parse HEAD)" > "$data/commits.txt"
for arm in "${!revisions[@]}"; do
    mkdir "$out/$arm"
    git archive "${revisions[$arm]}" src scripts | tar -x -C "$out/$arm"
    printf '%s=%s\n' "$arm" "${revisions[$arm]}" >> "$data/commits.txt"
done
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
    local limit=$1 deadline=$2 name=$3 rc=0 remaining=$(( $2 - SECONDS ))
    shift 3
    if (( remaining < limit )); then limit=$remaining; fi
    if (( limit > 0 )); then
        timeout --kill-after=5s "${limit}s" "$@" > "$data/$name.log" 2>&1 || rc=$?
    else
        rc=124
    fi
    printf '%s\t%s\t%s\n' "$name" "$rc" "$SECONDS" >> "$data/processes.tsv"
    return 0
}

config() {
    local arm=$1 root=$PWD
    if [[ -n ${revisions[$arm]:-} ]]; then root="$out/$arm"; fi
    export PYTHONPATH="$root/src:$root/scripts"
    export LNET_SCAN_W25A=off LNET_READER_W25B=off LNET_FUSED_STEM_CONV=off
    export LNET_VSCAN_MMA=off LNET_CAPTURE_SAFE_ORTHOGONAL=off LNET_READER_VEC_MODE=auto
    case "$arm" in
        wave25a|wave25ac|wave25abc) export LNET_SCAN_W25A=auto ;;
    esac
    case "$arm" in
        wave25b|wave25abc) export LNET_READER_W25B=auto ;;
    esac
    case "$arm" in
        wave25c|wave25ac|wave25abc) export LNET_FUSED_STEM_CONV=auto ;;
    esac
    case "$arm" in
        wave23a_mma) export LNET_VSCAN_MMA=on ;;
        wave23b_vec4) export LNET_READER_VEC_MODE=4 ;;
        wave22cv2_ps14) export LNET_CAPTURE_SAFE_ORTHOGONAL=on ;;
    esac
}

bench() {
    local limit=$1 deadline=$2 name=$3 command=$4
    shift 4
    run_capped "$limit" "$deadline" "$name" "$PYTHON_BIN" "$runner/audit.py" "$command" \
        --out "$data/$name.json" "$@"
}

train() (
    local limit=$1 deadline=$2 batch=$3 arm=$4 name=$5
    shift 5
    config "$arm"
    bench "$limit" "$deadline" "b${batch}_$name" train-arm --variant "$variant" --batch "$batch" "$@"
)

bench 20 390 device probe
"$PYTHON_BIN" "$runner/results.py" device "$data"
for arm in wave21c_base wave25a wave25b wave25c wave25ac wave25abc \
           wave23a_mma wave23b_vec4 wave22cv2_ps14; do
    train 310 3040 1024 "$arm" "$arm" --profile --full-kernels
done

best=$("$PYTHON_BIN" "$runner/results.py" select "$data")
if [[ -z $best ]]; then
    printf '%s\n' 'No eligible Wave 25 arm' > "$data/skip.txt"
    exit 0
fi
train 90 3220 1024 "$best" wave25_best_rev
train 90 3220 1024 wave21c_base wave21c_base_rev
if (( SECONDS < 3120 )); then
    (
        config "$best"
        bench 20 3480 best-probe probe
        bench 80 3480 dynamo dynamo
        bench 50 3480 fold infer-fold --variant "$variant" --batch 2
        tests_start=$SECONDS
        run_capped 250 3480 cuda-tests "$PYTHON_BIN" -m pytest -v -ra \
            tests/test_general_cuda_scans.py tests/test_factorized_reader_kernels.py \
            tests/test_fused_conv_stem_ln_gelu.py tests/test_p1_nr4.py tests/test_live_complex_rms.py
        pytest_rc=$(awk -F '\t' '$1 == "cuda-tests" {print $2}' "$data/processes.tsv")
        if [[ -f $data/cuda-tests.log ]]; then
            bench 15 3510 pytest wrap-pytest --log "$data/cuda-tests.log" --exit-code "$pytest_rc" \
                --elapsed "$((SECONDS - tests_start))"
        fi
        "$PYTHON_BIN" "$runner/results.py" regressions "$data"
    )
else
    printf '%s\n' 'Regression phase skipped at 52-minute threshold' >> "$data/skip.txt"
fi
if (( SECONDS < 3360 )); then
    for batch in 128 512 1024 2048; do
        for arm in wave21c_base "$best"; do
            # Keep paired measurements within the shared sweep deadline.
            if (( SECONDS >= 3580 )); then break 2; fi
            train 125 3705 "$batch" "$arm" "${arm}_sweep"
        done
    done
else
    printf '%s\n' 'Batch sweep skipped at 56-minute threshold' >> "$data/skip.txt"
fi
