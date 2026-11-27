# VisionAlphabet

This production branch contains the vision backbone and CUDA kernels in
`src/lnet/`, training and evaluation entry points in `scripts/`, the synthetic
benchmark in `h200/`, and CPU and CUDA regression tests in `tests/`.

## Automatic GPU settings

Run `PYTHONPATH=src:scripts python -m h200.bench_h200 --arm auto_lockin --out result.json`
or call `lnet.gpu_policy.apply_gpu_policy(device)` before constructing a model.
The ablation training launcher accepts `VA_AUTO_GPU_POLICY=1`. Explicit environment
values take precedence. A CPU device or unavailable CUDA leaves the environment unchanged.

| Compute capability | Reader | Horizontal scan | Hopper split flags |
| --- | --- | --- | --- |
| 9.0 and later (H100/H200 preset) | Dense NHWC | h55_lockin | Enabled |
| 8.9 (RTX 4090, L40S), 8.0 (A100) | Dense NHWC | bwd_opt | Omitted |
| Other pre-8.9 (including RTX 3080 Ti, 8.6) | Factorized | bwd_opt | Omitted |

The SM90a split operators additionally require an SM90 device. Fixed benchmark arms
and their `_rev` variants retain their original settings for reproduction.
Run the CPU suite with `CUDA_VISIBLE_DEVICES="" pytest -q tests/`.
