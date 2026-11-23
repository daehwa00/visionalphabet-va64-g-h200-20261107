# Wave 24C backbone probes

This directory contains isolated experiments based on Wave 21C
(`039440f2c0a9a9d006948fcbbf45dd5c96e36efb`). The model never imports them.

Run the following from the repository root with the project environment.
The convolution probe covers the ImageNet B=128 first stem stage. Padding
and im2col use the production LayerNorm/GELU epilogue. The implicit-convolution
probe optionally integrates that epilogue and uses recomputation plus the
existing LayerNorm and convolution VJPs for the training measurement.

```bash
export PYTHONPATH=src:scripts
probe=h200/wave24c_backbone_probe

CUDA_VISIBLE_DEVICES="" python "$probe/checkpoints.py"
CUDA_VISIBLE_DEVICES="" TRITON_ALWAYS_COMPILE=1 TRITON_DUMP_PTXAS_LOG=1 \
  python "$probe/offline.py"
CUDA_VISIBLE_DEVICES="" python "$probe/stem_nvrtc.py" --ptxas "$(command -v ptxas)"

for case in stem stem-autocast swiglu; do
  (flock -w 2400 9 && python "$probe/isolated.py" "$case") 9>/tmp/va-gpu.lock
done
for case in nr4 stem_geometry swiglu_block; do
  (flock -w 2400 9 && python "$probe/$case.py") 9>/tmp/va-gpu.lock
done
bash "$probe/verify.sh"
```

SwiGLU parity rows report mismatch count and maximum absolute difference in
the order eager forward, eager backward, compiled forward, compiled backward.
Convolution gradient differences are listed in weight, convolution bias,
normalization weight, normalization bias, and optional input order. The NR4
probe reports statistics, source gradient, and pole gradient differences.

The training driver records eight timed losses and eight pre-clip gradient
norms, CUDA profiler buckets, and exact allocator peaks. With a probe-only
patch, both arms execute the Wave 21C production path and measure run-to-run
variation.
