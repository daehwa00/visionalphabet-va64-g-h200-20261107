#!/usr/bin/env bash
set -euo pipefail
export VA_LOCKIN_RUNNER=h200.wave51_h200_lockin.run_wave51_h200
export VA_LOCKIN_TRANSPORT_TAG=VA_T183
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/va_t183_inductor_cache
export TRITON_CACHE_DIR=/tmp/va_t183_triton_cache
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../wave49_h200_lockin/run.sh"
