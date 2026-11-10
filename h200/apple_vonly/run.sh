#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ID="${APPLE_RUN_ID:-apple70m-vonly-100b-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
OUTPUT="/app/output/$RUN_ID"
SCRATCH="/app/scratch/$RUN_ID"
[[ ! -e "$OUTPUT" && ! -e "$SCRATCH" ]] || { echo 'Run path exists; refusing overwrite'; exit 2; }
mkdir -p "$OUTPUT" "$SCRATCH"
export PYTHONPATH="$ROOT/scripts" PYTHONUNBUFFERED=1
export VA64_G_JOB=apple70m-vonly VA64_G_CHECKPOINT_POLICY=local_only
exec python3 -u scripts/va64_h200_guard.py --root "$OUTPUT" --code-sha "$(git rev-parse HEAD)" \
 --relay-url https://apple70m-h200-20261109.gpupulse-monitor.workers.dev/va64g \
 --enroll-wait-seconds 120 --grace-seconds 30 --stall-seconds 1800 --poll-seconds 5 --metrics-optional \
 -- python3 -u h200/apple_vonly/job.py --root "$OUTPUT" --scratch "$SCRATCH"
