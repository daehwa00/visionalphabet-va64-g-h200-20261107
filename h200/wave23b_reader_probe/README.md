# Wave 23B reader probes

From the isolated checkout, prepare the historical source modules.

```bash
bash h200/wave23b_reader_probe/prepare.sh /tmp/va-t83-refs
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src:h200/wave17b_vec_reader_probe
python h200/wave23b_reader_probe/replay.py --refs /tmp/va-t83-refs --out /tmp/va-t83-plans
python h200/wave23b_reader_probe/replay.py --matrix --out /tmp/va-t83-matrix
```

Copy the checkout and the references to `/tmp/va-t83-gunsoo` and its `refs/`.
Extract separate git archives of Wave 17B and Wave 20B into the `wave17/` and
`wave20/` directories for model training comparisons. Run
`bash h200/wave23b_reader_probe/gunsoo.sh ablate` for forced vector widths,
`release` for the default dispatch comparison, `verify` for the regression
gates and `train` for both training orders and the controlled VRAM pair.
Each GPU process takes the shared lock. Each graph captures eight passes. Two
CUDA events bracket its replay after 24 rotated warmup samples per arm.

The graph report computes logical GB/s using one read and one write of
the packed activation and adjoint, with dweight including its reduction.
SASS counts are static instruction counts. Achieved logical GB/s is
derived from CUDA event time and does not measure physical HBM traffic.

Set `LNET_READER_VEC_MODE=wave20` for Wave 20B auto dispatch in the Wave 23B
checkout. Values `1`, `2`, `4` and `8` request those widths subject to
alignment, geometry and CUDA driver resource checks. The `4-rank` ablation
shares each weight-gradient adjoint tile across rank lanes. Use the historical
Wave 17B source for Wave 17B timings.
