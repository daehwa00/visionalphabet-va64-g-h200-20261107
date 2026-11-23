# Wave 22C CUDA Graph probes

These scripts leave the normal model builder and training path untouched. Set
`PYTHONPATH=src:scripts` from the repository root. Each training arm runs in
its own process with seed 501, BF16 autocast, batch 128, four warmup steps
and eight timed optimizer updates by default. cuDNN benchmarking is disabled.
For the shared RTX machine, hold the designated GPU lock across the sequence.

```sh
export PYTHONPATH=src:scripts OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
probe=h200/wave22c_cudagraph_probe
mkdir -p results
for arm in native safe safe native reduce-overhead graph graph reduce-overhead; do
    run=$((${run:-0} + 1))
    python "$probe/step.py" --arm "$arm" --output "results/$run.json"
done
python "$probe/step.py" --arm native --output results/native-traced.json \
    --trace results/native.trace.json
python "$probe/step.py" --arm safe --output results/safe-traced.json \
    --trace results/safe.trace.json
python "$probe/matrix.py" --device cuda --output results/matrix-cuda.json
CUDA_VISIBLE_DEVICES="" python "$probe/matrix.py" --output results/matrix-cpu.json
```

The timing loop reports monotonic raw wall time, CUDA event elapsed time, and
host time through submission before waiting for completion. The optional
trace captures a later step and excludes the profiler's explicit final
synchronization from the runtime API counts. GPU union duration merges
overlapping kernel, copy, and memset intervals. Profiler overhead and launch
overlap mean the trace GPU total and the steady-state wall time are distinct
measurements. Setup and steady-state memory peaks are reported separately.

The graph arm warms the same recipe on a side stream, then captures forward,
backward, clipping and fused AdamW. Setting the gradient attributes to
`None` is a host operation. It runs during capture so the graph records
writes into fresh gradient storage, and replay overwrites the same storage.
The probe retains the captured gradient references for a post-replay
finiteness check. It also checks the pre-clip norm on each timed update.

The matrix probe checks initialized weights and checkpoints from all three
builders, and compares dense and concentrated skew matrices at fixed norms.
Microbenchmark timing includes first-order differentiation and uses eager
execution. The polynomial variants are diagnostic candidates contained in
the probe.

Use `step.py --exp-fp32` to isolate the Taylor exponential from BF16
autocast while preserving it for the surrounding projection and model.
Use `gates.py --gate dynamo --output results/dynamo.json` for the existing
six-configuration Dynamo audit. For folding, use `gates.py --gate fold
--variant VARIANT --output results/fold.json` for each of the three models.
