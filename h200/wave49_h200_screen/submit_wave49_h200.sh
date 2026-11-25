#!/usr/bin/env bash
set -euo pipefail

result_dir=$(mktemp -d)
trap 'rm -rf -- "$result_dir"' EXIT
status=0
python3 h200/wave49_h200_screen/run_wave49_h200_screen.py --output "$result_dir" || status=$?
cd "$result_dir"
shopt -s nullglob
artifacts=(*.json)
if (( ${#artifacts[@]} )); then
    sha256sum -- "${artifacts[@]}"
    echo "VA_T169_BEGIN"
    tar -cf - -- "${artifacts[@]}" | gzip -9n | base64
    echo "VA_T169_END"
fi
exit "$status"
