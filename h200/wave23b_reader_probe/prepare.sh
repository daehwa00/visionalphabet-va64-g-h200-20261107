#!/usr/bin/env bash
set -euo pipefail
out="${1:?usage: prepare.sh REFERENCE_DIR}"
mkdir -p -- "$out"
for spec in 17:fda2db59 20:ad6fb7835a7a0272b2a70144ee2f70abd0763499; do
    version=${spec%:*}
    git show "${spec#*:}:src/lnet/factorized_reader_kernels.py" |
        sed "s/va_reader::/va_reader_w${version}::/g" > "$out/wave$version.py"
done
