#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
runner="$root/h200/wave51_type1_screen"
export PYTHONPATH="$root:$root/src:$root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
out="$(mktemp -d /tmp/va-t182-results.XXXXXX)"
trap 'rm -rf -- "$out"' EXIT
python_bin="${PYTHON_BIN:-python}"
"$python_bin" "$runner/compat.py" "$runner/metadata.py" > "$out/meta.json"
stage1=0
timeout 240s "$python_bin" -u "$runner/compat.py" scripts/bench_wave51_vscan_type1.py \
    --batch 1024 --warmup 2 --iters 6 --parity-batch 4 --output "$out/vscan_type1.json" || stage1=$?
stage2=0
timeout 30s "$python_bin" -u "$runner/compat.py" "$runner/rmsnorm_smoke.py" \
    "$out/rmsnorm_dynamo_smoke.json" || stage2=$?
printf '{"vscan_exit": %d, "rmsnorm_exit": %d}\n' "$stage1" "$stage2" > "$out/manifest.json"
cd "$out"
tar -czf wave51-type1.tar.gz -- ./*.json
sha256sum wave51-type1.tar.gz ./*.json
printf 'BEGIN_WAVE51_ARCHIVE\n'
base64 -w 160 wave51-type1.tar.gz
printf '\nEND_WAVE51_ARCHIVE\n'
