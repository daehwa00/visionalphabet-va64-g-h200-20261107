#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
SECRETS="${VA64_G_OWNER_SECRETS:-$HOME/.config/visionalphabet/va64-g-20261107/owner-secrets.json}"
[[ -f "$SECRETS" ]] || { echo 'Scoped owner secret file is missing'; exit 2; }
# This is the egress IP recorded by the previous H200 deployment. The new
# preflight must confirm it; a different IP stays blocked rather than broadening access.
EGRESS="${VA64_G_EGRESS_IPS:-203.253.146.151}"
CODE_SHA="$(git rev-parse HEAD)"
if [[ -n "${VA64_G_WRANGLER:-}" ]]; then
  WRANGLER=("$VA64_G_WRANGLER")
elif command -v wrangler >/dev/null; then
  WRANGLER=(wrangler)
else
  WRANGLER=(npx --yes wrangler@4.86.0)
fi
"${WRANGLER[@]}" deploy --config h200/va64/cloudflare/wrangler.jsonc --var "ALLOWED_EGRESS_IPS:$EGRESS" --var "EXPECTED_CODE_SHA:$CODE_SHA"
# The value travels only on stdin, never in the command or terminal output.
python - "$SECRETS" <<'PY' | "${WRANGLER[@]}" secret bulk --config h200/va64/cloudflare/wrangler.jsonc
from pathlib import Path
import sys,json
values=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({'IN10_OWNER_TOKEN':values['IN10_OWNER_TOKEN']}))
PY
