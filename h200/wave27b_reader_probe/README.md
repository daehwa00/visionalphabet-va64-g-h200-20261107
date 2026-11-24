# Wave 27B reader probe

Use `LNET_READER_W25B=auto LNET_READER_VEC=auto` for the Wave 26 baseline.
Run the Whole H200 bundle at B=1024 with these values of `LNET_READER_W27B`.

| Value | Dispatch changes relative to Wave 25B |
|---|---|
| `off` (default) | Baseline |
| `dweight_2cta` | 56² vector-2 dweight, 14 rows, 4 warps, 132 workers on Whole H200 |
| `scalar_hiocc` | 56² scalar forward/dinput, 8 rows, 4 warps. 28² scalar dinput, 4 rows, 4 warps |
| `full_hopper` | Same spatial candidates plus vector-2 14² spatial. 56²/28² vector-2 dweight with 176/132 workers and scalar 14² dweight with 7 rows on Whole H200 |
| `auto` | `full_hopper` on SM 9.0. Baseline elsewhere |

Explicit presets also work on SM 8.6 for auxiliary checks. The candidates are
limited to specialized rank-4, 160-mode, 3×3 real-reader cases with homogeneous
BF16 or FP16 operands. The CUDA driver checks residency and local memory before
a candidate is used.

The offline replay uses the production dispatch, CUDA 12.8 ptxas, and cuobjdump.
To run it from the repository root on a CPU host:

```sh
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe:h200/wave25b_reader_probe
git show a670a752:src/lnet/factorized_reader_kernels.py >/tmp/va-t95-base.py
python h200/wave27b_reader_probe/offline_matrix.py \
  --baseline /tmp/va-t95-base.py --out /tmp/va-t95-matrix.jsonl
```
