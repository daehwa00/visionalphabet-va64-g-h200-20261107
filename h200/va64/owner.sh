#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
SECRETS="${VA64_G_OWNER_SECRETS:-$HOME/.config/visionalphabet/va64-g-20261107/owner-secrets.json}"
ROOT="${VA64_G_OBSERVER_ROOT:-$HOME/experiments/visionalphabet-va64-g-observer-20261107}"
[[ -f "$SECRETS" ]] || { echo 'Scoped owner secret file is missing'; exit 2; }
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export VA64_G_RELAY_URL=https://visionalphabet-va64-g-20261107.gpupulse-monitor.workers.dev/va64g
export VA64_G_WANDB_PROJECT=visionalphabet-rebuttal-va64 VA64_G_WANDB_ENTITY=daehwa
exec python -u scripts/va64_h200_observer.py --root "$ROOT" --secrets "$SECRETS" --poll-seconds 15
