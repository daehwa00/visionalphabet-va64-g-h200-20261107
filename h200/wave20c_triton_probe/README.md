# Wave 20C training-step audit

Run from the repository root using the provisioned CUDA PyTorch environment.
Set `PYTHONPATH=src:scripts:.` and `TORCH_COMPILE_DEBUG=1`. Use a fresh
`TORCHINDUCTOR_CACHE_DIR` for the first capture so both AOT graphs are dumped.

```sh
python -m h200.wave20c_triton_probe.audit_step --cudnn-benchmark --out audit.json
python -m h200.wave20c_triton_probe.map_kernels \
    --profile audit.json --debug-dir torch_compile_debug --out attribution.json
```

The default workload is P1/NR4, B=128, synthetic 224×224 inputs, BF16 AMP,
and `torch.compile(mode="default")`. Omit `--cudnn-benchmark` for controlled
allocator measurements. The probe uses the existing Wave 17B training-arm
optimizer, clipping, and warm-up. Its Chrome trace includes CPU and CUDA
events. CUDA totals sum actual kernel and device memory events, excluding
profiler annotation spans. The two NVRTC scan directions share a symbol
and appear together in `general_scan_shared`.

Point the mapper at a single debug run for one compiled arm. The mapping
matches emitted Triton functions with the contributing post-decomposition
FX nodes and their source comments. A single emitted function can be reused
at several graph launch sites. Handwritten Triton launches may have no
matching emitted function and require source inspection.
