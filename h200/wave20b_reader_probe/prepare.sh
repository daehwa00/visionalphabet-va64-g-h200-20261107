#!/usr/bin/env bash
set -euo pipefail

out=$(realpath -m -- "${1:?usage: prepare.sh REFERENCE_DIR}")
mkdir -p -- "$out"
for spec in 14:42ff1ee 17:fda2db59 19:595c9d3; do
    version=${spec%:*}
    sha=${spec#*:}
    git show "$sha:src/lnet/factorized_reader_kernels.py" |
        sed "s/va_reader::/va_reader_w${version}::/g" > "$out/wave$version.py"
done
