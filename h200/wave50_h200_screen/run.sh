#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root:$root/src:$root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
exec "${PYTHON_BIN:-python}" -m h200.wave50_h200_screen.run_screen "$@"
