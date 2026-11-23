#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
out=$(realpath -m -- "${1:?usage: verify.sh OUTPUT_DIR [BATCH] [PYTHON]}")
batch=${2:-1024}
py=${3:-python3}
mkdir -p -- "$out"
root=$PWD
variant=17-p1-nr4-1024d-v3
baseline=42ff1ee5170892b94d44a4a17106af28b71400ab
bench=h200/liverms_v2_opt/bench_h200.py
probe=h200/wave17b_vec_reader_probe/bench.py

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
git archive "$baseline" | tar -x -C "$tmp"

run_candidate() {
    PYTHONPATH="$root/src:$root/scripts" "$py" "$@"
}

LNET_READER_VEC=1 PYTHONPATH="$tmp/src:$tmp/scripts" "$py" "$bench" train-arm \
    --variant "$variant" --batch "$batch" --out "$out/train-baseline.json"
for arm in 1 2 4 auto; do
    LNET_READER_VEC=$arm run_candidate "$bench" train-arm \
        --variant "$variant" --batch "$batch" --out "$out/train-$arm.json"
done
run_candidate "$probe" bench --batch "$batch" \
    --baseline "$tmp/src/lnet/factorized_reader_kernels.py" --out "$out/reader.jsonl"
run_candidate "$probe" parity \
    --baseline "$tmp/src/lnet/factorized_reader_kernels.py" --out "$out/parity.jsonl"
LNET_READER_VEC=auto run_candidate "$bench" dynamo --out "$out/dynamo.json"
LNET_READER_VEC=auto run_candidate "$bench" infer-fold \
    --variant "$variant" --batch 2 --out "$out/fold.json"
