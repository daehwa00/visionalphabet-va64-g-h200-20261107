#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ID="apple70m-residual-resume-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT="/app/output/$RUN_ID"
SCRATCH="/app/scratch/$RUN_ID"
[[ ! -e "$OUTPUT" && ! -e "$SCRATCH" ]] || exit 2
mkdir -p "$OUTPUT" "$SCRATCH"
export PYTHONPATH="$ROOT/scripts" PYTHONUNBUFFERED=1
export VA64_G_JOB=apple70m-train VA64_G_CHECKPOINT_POLICY=local_only
exec python3 -u h200/apple_checkpoint/guard.py --root "$OUTPUT" --code-sha "$(git rev-parse HEAD)" \
 --relay-url https://apple70m-h200-20261109.gpupulse-monitor.workers.dev/va64g \
 --share-token-with-child --enroll-wait-seconds 120 --grace-seconds 30 --stall-seconds 1800 --poll-seconds 5 --metrics-optional \
 -- python3 -u h200/apple_resume/job.py --root "$OUTPUT" --scratch "$SCRATCH"
