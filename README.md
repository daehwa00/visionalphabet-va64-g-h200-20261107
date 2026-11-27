# VisionAlphabet

This production branch contains the vision backbone and CUDA kernels in
`src/alphabet/`, training and evaluation entry points in `scripts/`, the synthetic
benchmark in `h200/`, and CPU and CUDA regression tests in `tests/`.

## Automatic GPU settings

Run `PYTHONPATH=src:scripts python -m h200.bench_h200 --arm auto --out result.json`
or call `alphabet.gpu_policy.apply_gpu_policy(device)` before constructing a model.
The ablation training launcher accepts `ALPHABET_AUTO_GPU_POLICY=1`. Explicit environment
values take precedence. A CPU device or unavailable CUDA leaves the environment unchanged.

| Compute capability | Reader | Horizontal scan | Hopper split flags |
| --- | --- | --- | --- |
| 9.0 (H100/H200) | Dense NHWC | hopper_specialized | BF16 split |
| Later than 9.0 (Blackwell) | Dense NHWC | optimized_backward | Omitted |
| 8.9 (RTX 4090, L40S), 8.0 (A100) | Dense NHWC | optimized_backward | Omitted |
| Other pre-8.9 (including RTX 3080 Ti, 8.6) | Factorized | optimized_backward | Omitted |

On Hopper, `--arm auto` defaults to BF16 split (~137 ms on H200, cosine
0.99997–1.00000 against FP32). Set `ALPHABET_ALLOW_FP8=1` or select
`--arm hopper_fp8_resident` for FP8 split with resident weights (~123.3–123.7 ms).
The SM90a split operators additionally require an SM90 device. Fixed benchmark arms
are `reference`, `bf16_compiled`, `hopper_bf16_split`, `hopper_fp8_split`, and
`hopper_fp8_resident`. Append `_rev` to select the reverse-order arm.
Run the CPU suite with `CUDA_VISIBLE_DEVICES="" pytest -q tests/`.
