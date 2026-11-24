# Wave 28B reader probe

Whole H200 runs use `LNET_READER_W25B=auto LNET_READER_W27B=off LNET_READER_VEC=auto`.
Compare the following values of `LNET_READER_W28B` at B=1024.

| Value | Dispatch on Whole H200 |
|---|---|
| `off` (default) | Existing Wave 27B/Wave 25B dispatch |
| `keep_fwd_hopper` | Wave 26 forward at 56²/28², Wave 27B full_hopper forward at 14², and full_hopper backward |
| `dweight_3cta` | Same as keep_fwd_hopper with 4 rows, 4 warps, and 132 dweight workers at 56² |
| `bwd_hiocc_v2` | Same as keep_fwd_hopper with scalar 7-row dinput at 56²/14², and a smaller rank/row tile for 14² dweight |
| `fwd_vec2_opt` | Same as keep_fwd_hopper with 4-row vec2 forward at 56²/28² and 4/2 warps respectively |
| `auto` | keep_fwd_hopper on SM 9.0 and prior dispatch elsewhere |

Wave 28B takes precedence for eligible launches. Explicit presets also run on SM 8.6
for numerical checks. Eligibility requires specialized 56², 28², or 14² real reader
with 160 modes, rank 4, 3×3 kernels, and homogeneous BF16 or FP16 operands.

From the repository root:

```sh
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe:h200/wave25b_reader_probe
git show a670a752:src/lnet/factorized_reader_kernels.py >/tmp/va-wave26-reader.py
CUDA_VISIBLE_DEVICES="" python h200/wave28b_reader_probe/offline_matrix.py \
  --baseline /tmp/va-wave26-reader.py --out /tmp/va-w28b-ptxas.jsonl
```

For CUDA parity, run on a GPU host. The probe executes numerical comparisons only.

```sh
python h200/wave28b_reader_probe/parity.py \
  --baseline /tmp/va-wave26-reader.py --out /tmp/va-w28b-parity.jsonl
```
