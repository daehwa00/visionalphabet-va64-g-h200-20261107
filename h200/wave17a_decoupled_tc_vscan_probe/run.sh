#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

if [[ $# != 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi
mkdir -p -- "$1"
out=$(realpath -- "$1")
python_bin=${WAVE17A_PYTHON:-python3}
suite=h200/wave17a_decoupled_tc_vscan_probe
export PYTHONPATH="$PWD/src:$PWD/scripts"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
batch=${WAVE17A_BATCH:-1024}
for policy in off auto; do
    "$python_bin" "$suite/model.py" train-arm --batch "$batch" --policy "$policy" \
        --profile --out "$out/train_$policy.json"
done
