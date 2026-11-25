#!/usr/bin/env bash
set -euo pipefail

result_dir=$(mktemp -d)
trap 'rm -rf -- "$result_dir"' EXIT
status=0
python3 h200/wave49_h200_probe2/run_wave49_probe2.py --output "$result_dir" || status=$?
cd "$result_dir"
shopt -s nullglob
for artifact in *.json; do
    sha256sum -- "$artifact"
    echo "VA_T172_BEGIN $artifact"
    gzip -9nc -- "$artifact" | base64
    echo "VA_T172_END $artifact"
done
exit "$status"
