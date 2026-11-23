# Wave 21B stem probe

The branch retains Wave 17B production kernels. The patch tests storing all
affine partials in a CTA's shared memory and reducing one channel per thread,
while preserving both the worker count and summation order.

Apply the candidate patch in a disposable checkout before running A/B.

```sh
git apply h200/wave21b_stem_probe/parallel-reduction.patch
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=src python h200/wave21b_stem_probe/offline.py /tmp/wave21b-cubins --candidate
```

The GPU scripts take the same shared GPU lock used by other benchmarks on
`gunsoo`. Run the training script on the GPU host with the candidate applied.

```sh
bash h200/wave21b_stem_probe/verify.sh "$HOME/miniconda3/envs/alphabet/bin/python" results
(
  flock -w 2400 9
  PYTHONPATH=src:scripts "$HOME/miniconda3/envs/alphabet/bin/python" h200/wave21b_stem_probe/isolated.py
) 9>/tmp/va-gpu.lock
(
  flock -w 2400 9
  PYTHONPATH=src:scripts "$HOME/miniconda3/envs/alphabet/bin/python" h200/wave21b_stem_probe/adjacent.py
) 9>/tmp/va-gpu.lock
```
