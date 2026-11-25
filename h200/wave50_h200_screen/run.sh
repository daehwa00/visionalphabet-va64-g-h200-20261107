#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root:$root/src:$root/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -m h200.wave50_h200_screen.run_wave50_screen "$@"
