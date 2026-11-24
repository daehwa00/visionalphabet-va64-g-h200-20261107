# Wave 28A scan presets

| `LNET_SCAN_W28A` | Dispatch |
| --- | --- |
| `off` (default) | Follow `LNET_SCAN_W27A`. Set Wave 27A to `wave_plan` for the #810 control. |
| `wp_all_waves` | Trim partial CTA waves on SM 9.0 when the maximum number of lines per worker stays constant. |
| `wp_batch2` | Compose the Wave 27A wave plan with two-cell weight reuse for 56×56 vertical backward. |
| `wp_batch2_all`, `auto` | Extend `wp_batch2` to 28×28, 14×14 and 7×7 vertical backward. An odd final cell uses the scalar adjoint. |
| `wp_vscan_opt` | Combine the Wave 27A wave plan with 14-row FP32 checkpoints for 56×56 vertical backward. |

Use the same Wave 26 production environment across the H200 arms. The Wave 28A presets select their own Wave 27A behavior. The batched and checkpoint candidates require the same CTA residency as the original kernel and no local memory.

Replay the #810 `w27a_wave_plan.json` profile with CUDA 12.8 and the GPU hidden.

```sh
export CUDA_VISIBLE_DEVICES=""
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64
export PYTHONPATH=src:h200/wave25a_scan_probe:h200/wave27a_scan_probe
python h200/wave28a_scan_probe/replay_h200_wave28a.py /path/to/w27a_wave_plan.json /path/to/results
```

Run numerical validation on the auxiliary GPU under its shared lock. The probe also selects one-warp source-cached kernels to exercise the backward candidates on SM 8.6.

```sh
export PYTHONPATH=src:scripts:tests OMP_NUM_THREADS=4
(
    flock -w 2400 9 || exit
    python h200/wave28a_scan_probe/validate.py parity.jsonl
    LNET_SCAN_W28A=wp_batch2_all python -m pytest -q tests/test_general_cuda_scans.py
) 9>/tmp/va-gpu.lock
```
