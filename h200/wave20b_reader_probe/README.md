# Wave 20B reader probes

From the repository root, prepare the historical reader sources. The aliases
give the historical custom ops distinct registration namespaces.

```bash
bash h200/wave20b_reader_probe/prepare.sh /tmp/va-t74-refs
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
python h200/wave20b_reader_probe/shapes.py
python h200/wave20b_reader_probe/replay.py --refs /tmp/va-t74-refs --out /tmp/va-t74-replay
python h200/wave20b_reader_probe/replay.py --matrix --out /tmp/va-t74-matrix
```

The CPU shape probe follows the real model topology and uses BF16 linear
projections with a stand-in for the spatial convolution. The CUDA probe
intercepts the actual spatial call. In the plan replay, NVRTC compiles the
same variants used by production dispatch, ptxas reports resource usage and
a launch interceptor records the exact grids. Occupancy is estimated from
the specified SM limits and the assembled resource counts.

Stage the repository into `/tmp/va-t74-gunsoo`, copy the references into its
`refs/`, and extract the Wave 17B git archive into its `wave17/` before using
`gunsoo.sh`. Each command that uses the GPU acquires `/tmp/va-gpu.lock` for its
own process. The training runner performs both arm orders and a cuDNN
autotuning-disabled control. The general screening runner records the actual
common batch after accounting for free memory.
