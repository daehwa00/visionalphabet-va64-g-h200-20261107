#!/usr/bin/env bash
# Create a per-experiment receiver in the allocated user's output directory.
set -eu -o pipefail
umask 077
prior=/app/output/cand12-layout-deploy-20261118/observed.json
if [[ -f $prior ]]; then echo PREVIOUS_SCOPE_VISIBLE=yes; else echo PREVIOUS_SCOPE_VISIBLE=no; fi
root=/app/output/va-cand12-matched-stage-20261118
mkdir "$root"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$root/receiver.pem" 2>/dev/null
openssl pkey -in "$root/receiver.pem" -pubout
