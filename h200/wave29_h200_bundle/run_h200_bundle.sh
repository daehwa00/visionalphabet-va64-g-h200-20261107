#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
exec 3>&1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
runner="$PWD/h200/wave29_h200_bundle"
out=$(mktemp -d /app/output/va-t105-h200.XXXXXX)
data="$out/results"
mkdir "$data"
export LC_ALL=C PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$out/inductor" VA_T93_BENCH="$out/bench_h200.py"
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED LNET_READER_VEC_MODE
export LNET_SCAN_W25A=auto LNET_READER_W25B=auto LNET_READER_VEC=auto LNET_FUSED_STEM_CONV=auto
export LNET_CAPTURE_SAFE_ORTHOGONAL=on LNET_VSCAN_MMA=off
export LNET_SCAN_W27A=wave_plan LNET_READER_W27B=off LNET_STEM_W27C=off
export LNET_SCAN_W28A=wp_batch2_all LNET_READER_W28B=bwd_hiocc_v2 LNET_MODEL_W28C=combined
export LNET_SCAN_W29A=off LNET_READER_W29B=off LNET_MODEL_W29C=off
export VA_T105_TRACE="$out/profile.json"
exec 2>"$data/runner.log"

finish() {
    local rc=$? archive_rc=0
    trap - EXIT
    exec >&3 2>&3
    set +e
    printf '%d\n' "$rc" > "$data/exit-code.txt"
    printf '%d\n' "$SECONDS" > "$data/elapsed-seconds.txt"
    if (( rc != 0 )) && [[ -s $data/environment.log ]]; then
        printf 'Environment log retained in archive\n'
    fi
    timeout --kill-after=5s 360s python3 "$runner/results.py" transport "$data" "$out" > "$out/transport.log" 2>&1
    archive_rc=$?
    if (( archive_rc != 0 )); then
        tail -c 512 "$out/transport.log"
    fi
    if (( archive_rc == 0 )); then
        sha256sum "$out/task105.tar.xz" | awk '{print "VA_T105_SHA256 " $1}'
        base64 -w 240 "$out/task105.tar.xz" | sed 's/^/VA_T105_XZ /'
        printf 'VA_T105_END\n'
    fi
    printf 'Task 105 result %s, runner exit %d, transport exit %d\n' "$out" "$rc" "$archive_rc"
    (( rc == 0 )) || exit "$rc"
    exit "$archive_rc"
}
trap finish EXIT
trap 'exit 124' TERM
variant=17-p1-nr4-1024d-v3
ref=5ec4d31eb4c8e3075af24739db9d10ace2fd1c27
git show "$ref:h200/liverms_v2_opt/bench_h200.py" > "$VA_T93_BENCH"
printf 'benchmark=%s\nbaseline=%s\nbundle=%s\n' "$ref" a670a752a01ecf32e90b6016777750569e422148 "$(git rev-parse HEAD)" > "$data/commits.txt"
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
    /app/output/va-t92-h200.aOb6kB/environment/venv/bin/python \
    /app/output/va-t93-h200.TrMSFC/environment/venv/bin/python \
    /app/output/va-t93-h200.XMqpcl/environment/venv/bin/python \
    /app/output/va-t101-h200.*/environment/venv/bin/python \
    /app/output/va-t105-h200.*/environment/venv/bin/python \
    /app/output/va-t97-h200.*/environment/venv/bin/python \
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
    # Cold downloads are several GiB and have exceeded the former 240s cap.
    export UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=4 UV_CONCURRENT_DOWNLOADS=6
    timeout --kill-after=5s 900s env PYTHONPATH="$setup/bootstrap" python3 -m uv pip install --python "$PYTHON_BIN" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$PYTHON_BIN"
fi > "$data/environment.log" 2>&1


printf 'Python environment ready after %ss\n' "$SECONDS"
export PYTHONPATH="$PWD/src:$PWD/scripts"

run_capped() {
    local limit=$1 name=$2 rc=0
    shift 2
    # Leave ten seconds for timeout escalation before the packaging boundary.
    local remaining=$((58 * 60 - SECONDS - 10))
    if (( remaining <= 0 )); then
        rc=125
    else
        if (( limit > remaining )); then limit=$remaining; fi
        timeout --kill-after=10s "${limit}s" "$@" > "$data/$name.log" 2>&1 || rc=$?
    fi
    printf '%s\t%s\t%s\n' "$name" "$rc" "$SECONDS" >> "$data/processes.tsv"
    printf '%s rc=%s elapsed=%ss\n' "$name" "$rc" "$SECONDS"
}

bench() {
    local limit=$1 name=$2 command=$3
    shift 3
    run_capped "$limit" "$name" "$PYTHON_BIN" "$runner/audit.py" "$command" --out "$data/$name.json" "$@"
}

set_combo() {
    export LNET_SCAN_W29A=$1 LNET_READER_W29B=$2 LNET_MODEL_W29C=$3
}

train() (
    name=$1
    shift
    set_combo "$@"
    bench 300 "$name" train-arm --variant "$variant" --batch 1024 --profile --full-kernels
)

pytest_run() {
    local limit=$1 name=$2 test_start=$SECONDS test_rc
    shift 2
    run_capped "$limit" "$name" "$PYTHON_BIN" -m pytest -q -ra "$@"
    read -r _ test_rc _ < <(tail -n 1 "$data/processes.tsv")
    if [[ -f $data/$name.log ]]; then
        "$PYTHON_BIN" "$runner/audit.py" wrap-pytest --log "$data/$name.log" \
            --out "$data/pytest_$name.json" --exit-code "$test_rc" --elapsed "$((SECONDS - test_start))"
    fi
}

reader_parity() (
    git show a670a752:src/lnet/factorized_reader_kernels.py > "$out/wave26_reader.py"
    export PYTHONPATH="$PYTHONPATH:$PWD/h200/wave17b_vec_reader_probe:$PWD/h200/wave25b_reader_probe"
    run_capped 180 parity_b "$PYTHON_BIN" "$runner/reader_parity.py" \
        --baseline "$out/wave26_reader.py" --out "$out/parity_b.jsonl"
    "$PYTHON_BIN" "$runner/results.py" reader-parity "$data" "$out/parity_b.jsonl"
)

scan_parity() {
    run_capped 180 parity_a env PYTHONPATH="$PYTHONPATH:$PWD/tests:$PWD/h200/wave28a_scan_probe" \
        "$PYTHON_BIN" "$runner/scan_parity.py" "$data/parity_a.json" "$LNET_SCAN_W29A" "${1:-0}"
}

model_probe() {
    run_capped 480 model_probe env PYTHON="$PYTHON_BIN" bash \
        h200/wave29c_model_probe/run.sh "$data" verify
}

regression() {
    if (( SECONDS < 55 * 60 )); then
        [[ $LNET_SCAN_W29A = off ]] || scan_parity
        reader_parity
        pytest_run 180 parity_c tests/test_model_w28c.py tests/test_model_w29c.py
        bench 240 dynamo dynamo
        bench 240 fold infer-fold --variant "$variant" --batch 2
        [[ $LNET_MODEL_W29C = off ]] || model_probe
    fi
}

shard_pytest() {
    local track=$1 first=$2 size=$3 slots=$4 test_file
    case "$track" in
        a) test_file=tests/test_general_cuda_scans.py ;;
        b) test_file=tests/test_factorized_reader_kernels.py ;;
        c) test_file=tests/test_model_w28c.py ;;
        c29) test_file=tests/test_model_w29c.py ;;
        *) exit 2 ;;
    esac
    [[ $first =~ ^[0-9]+$ && $size =~ ^[1-9][0-9]*$ && $slots =~ ^[1-9][0-9]*$ ]] || exit 2
    run_capped 60 collect "$PYTHON_BIN" -m pytest --collect-only -q "$test_file"
    mapfile -t nodes < <(awk -v prefix="$test_file::" 'index($0, prefix) == 1' "$data/collect.log")
    (( ${#nodes[@]} > first )) || exit 2
    printf '%s\t%s\t%s\t%s\t%s\n' "$track" "${#nodes[@]}" "$first" "$size" "$slots" > "$data/shard-layout.tsv"
    printf '%s\n' "${nodes[@]}" > "$data/pytest-nodes.txt"
    for ((i=first; i<${#nodes[@]} && i<first+size*slots; i+=size)); do
        pytest_run 300 "${track}_$i" "${nodes[@]:i:size}"
    done
}

bench 90 device probe
"$PYTHON_BIN" "$runner/results.py" device "$data"

if [[ ${1:-} == pytest-shard || ${1:-} == regression ]]; then
    mode=$1
    shift
    set_combo "$1" "$2" "$3"
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" > "$data/active-presets.tsv"
    shift 3
    if [[ $mode == pytest-shard ]]; then
        shard_pytest "${@:1:4}"
        if [[ -n ${5:-} ]]; then
            [[ $5 =~ ^[0-9]+$ ]] && (( $5 < 32 )) || exit 2
            scan_parity "$5"
        fi
    else
        regression
    fi
    exit 0
fi
while read -r arm a b c; do
    train "$arm" "$a" "$b" "$c"
done < <("$PYTHON_BIN" "$runner/results.py" arms)
winners=$("$PYTHON_BIN" "$runner/results.py" select "$data")
read -r a b c <<< "$winners"
train w29_best_combo "$a" "$b" "$c"
train w29_best_combo_rev "$a" "$b" "$c"
train w28_base_rev off off off
set_combo "$a" "$b" "$c"
regression

if (( SECONDS < 55 * 60 )) && [[ $b != off ]]; then
    pytest_run 300 b tests/test_factorized_reader_kernels.py
fi
if (( SECONDS < 55 * 60 )) && [[ $a != off ]]; then
    pytest_run 300 a tests/test_general_cuda_scans.py
fi
"$PYTHON_BIN" "$runner/results.py" summarize "$data"
