# Wave 27A scan presets

Set `LNET_SCAN_W27A` per process. The default is `off`. Keep the Wave 26 production environment for each H200 arm and vary this setting.

| Value | Scan change |
| --- | --- |
| `off` | Wave 26 dispatch |
| `auto`, `wave_plan` | Trim a partial CTA wave from eligible vertical backward launches when the longest worker loop keeps its iteration count |
| `bwd_batch2` | Reuse each MLP weight load for two adjacent cells in the 56×56 vertical backward adjoint |
| `bwd_chunk` | Keep a full FP32 trajectory for eligible 56×56 vertical backward launches |
| `hscan_unroll` | Apply eight-way unroll to dense cached four-warp 56×56 horizontal backward and cached three-warp horizontal forward |

Replay an H200 profile using CUDA 12.8 with the GPU hidden.

```sh
export CUDA_VISIBLE_DEVICES=""
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64
export PYTHONPATH=src:h200/wave25a_scan_probe
python h200/wave27a_scan_probe/replay.py /path/to/b1024_best_rev.json /path/to/results
```

On the auxiliary GPU, use the required GPU lock for parity and isolated graph timing. Graph timing uses two-warp vertical geometry and the packed FP32 weight cache from Hopper. This lets both full trajectories fit on SM 8.6. It also uses the production warp widths for the horizontal candidates.

```sh
export PYTHONPATH=src:scripts:tests OMP_NUM_THREADS=4
(
    flock -w 2400 9 || exit
    python h200/wave27a_scan_probe/validate.py parity parity.jsonl
    python h200/wave27a_scan_probe/validate.py graphs graphs.jsonl
) 9>/tmp/va-gpu.lock
```
