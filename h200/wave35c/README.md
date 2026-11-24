# Wave 35C H200 A/B

Keep the Wave 33 environment constant across arms. Set `LNET_MODEL_W35C`
before constructing and compiling the model. Use a fresh process for each arm.

| Mode | Experiment |
|---|---|
| `off` | Baseline. Default when the variable is unset. |
| `auto` | Baseline while the H200 speed and accuracy gates are pending. |
| `stem_v2` | RGB strip gathering for channels-last Stage 0, smaller backward tiles, and a 64-channel Tensor Core arm. NCHW uses the direct patch gather. |
| `swiglu_v3` | Divisor-tiled SwiGLU with 1024 values per CTA for widths divisible by eight. |
| `rmsnorm_bwd_inductor` | Existing ATen RMSNorm forward and compiler-visible tensor VJP. |
| `inductor_tune` | Scoped epilogue and aggressive fusion options. |
| `full_w35c` | All four arms. |

The tuning context covers the first call as well as subsequent calls that might
trigger recompilation.

```python
from lnet.model_w35c import apply_inductor_tune

with apply_inductor_tune():
    compiled = torch.compile(model, mode="default")
    # Keep warmup, forward, backward, and timing inside the context.
    benchmark(compiled)
```

For a native SwiGLU comparison, use Wave 35C off and set the Wave 29C
preset to `rmsnorm_bwd`. This preserves the custom RMSNorm backward while
selecting native SwiGLU with the default Wave 28C policy.

```sh
export CUDA_VISIBLE_DEVICES=""
PYTHONPATH=src python h200/wave35c/audit.py > /tmp/w35c-triton.csv
PYTHONPATH=src python h200/wave35c/nvrtc_audit.py > /tmp/w35c-nvrtc.csv
PYTHONPATH=src python h200/wave35c/inductor_cpu.py
PYTHONPATH=src:scripts python h200/wave35c/verify_cpu.py /path/to/base-checkpoints
```

Generate base checkpoints by running
`h200/wave25c_stem_probe/checkpoints.py /path/to/base-checkpoints`
with `PYTHONPATH=src:scripts` from commit `41eb6b30463110525f07a26da614e683a8fe34bc`.

The resource CSVs report static SASS metrics and upper bounds for resident warps.
The load and store widths are the largest widths anywhere in each kernel.
They are therefore compatible with scalar loads for a specific operand, such as
RGB patch entries. For CO64 Stage 0 the active original path is NVRTC.
The untuned CO64 Triton rows are a diagnostic comparison of the legacy
Tensor Core schedule at that width.

On CUDA, run `pytest -q tests/test_model_w35c.py` to check the experiment arms.
The CUDA tests and full-step timings require the H200 run.
