"""Time fused and decomposed vertical scan backward on H200."""

# Diagnostic scripts use private benchmark hooks and print live progress.
# ruff: noqa: ANN001, ANN201, ANN202, SLF001, T201
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from bench_scan_micro import make_inputs  # noqa: E402
from probe_readproj_graph import l2_buffer, measure  # noqa: E402

from lnet import general_cuda_scans as scans  # noqa: E402

BATCH = 1024
MODES = 160
CHUNKS = (4, 8, 16, 32, 64, 128)


def path_backward(features, gr, gi, path):
    wi, bi, wo, _ = path
    p = features.permute(2, 1, 0).contiguous()
    g = torch.stack((gr, gi), dim=-1).view(-1, MODES, 2).transpose(0, 1).contiguous()
    z = torch.bmm(p, wi.transpose(1, 2), out_dtype=torch.float32)
    z.add_(bi.unsqueeze(1))
    # SiLU derivative is evaluated in FP32 before the BF16 GEMMs.
    sigmoid = z.sigmoid()
    derivative = (sigmoid * (1 + z * (1 - sigmoid))).to(p.dtype)
    hidden = (z * sigmoid).to(p.dtype)
    dh = torch.bmm(g, wo)
    dz = dh * derivative
    dp = torch.bmm(dz, wi).permute(2, 1, 0).contiguous()
    dwi = torch.bmm(dz.transpose(1, 2), p, out_dtype=torch.float32)
    dwo = torch.bmm(g.transpose(1, 2), hidden, out_dtype=torch.float32)
    return dp, (dwi, dz.sum(dim=1, dtype=torch.float32), dwo, g.sum(dim=1, dtype=torch.float32))


def chunk_ops(inputs, chunk, evict):  # noqa: C901
    pole, states, gains, path, gr, gi = inputs
    _, batch, h, w, m = states.shape
    slices = [slice(i, i + chunk) for i in range(0, batch, chunk)]
    scratch = states.new_empty((4, *states.shape[1:]))
    seed = states.new_full((8, chunk * h * w, m), 0.03125)
    pole_fp32 = [p.float() for p in pole]

    def scan_forward(part):
        return scans._vscan(
            pole, states[:, part], gains, path, coarse=False, epsilon=1e-6, precise=False, pure=True
        )

    def scan_backward(part, dp):
        return scans._vscan(
            pole_fp32,
            states[:, part],
            gains,
            path,
            coarse=False,
            epsilon=1e-6,
            precise=False,
            features=dp,
            state_grad=scratch[:, part],
            pure=True,
        )

    def forward():
        for part in slices:
            scan_forward(part)

    def backward():
        totals = [torch.zeros_like(p) for p in pole_fp32]
        for part in slices:
            for dst, src in zip(totals, scan_backward(part, seed)[:4], strict=True):
                dst.add_(src)

    def path_only():
        totals = [torch.zeros_like(p, dtype=torch.float32) for p in path]
        for part in slices:
            _, grads = path_backward(seed, gr[part], gi[part], path)
            for dst, src in zip(totals, grads, strict=True):
                dst.add_(src)

    def combined():
        totals = [torch.zeros_like(p, dtype=torch.float32) for p in (*pole_fp32, *path)]
        for part in slices:
            dp, grads = path_backward(scan_forward(part), gr[part], gi[part], path)
            for dst, src in zip(totals, (*scan_backward(part, dp)[:4], *grads), strict=True):
                dst.add_(src)
        return (*totals[:4], scratch, *totals[4:])

    result = {}
    ops = [("pure_fwd_ms", forward), ("pure_bwd_ms", backward), ("path_bwd_ms", path_only)]
    if chunk != batch:
        ops.append(("combined_ms", combined))
    for op, fn in ops:
        print(f"  chunk={chunk} {op}", flush=True)
        result[op] = measure(fn, evict, repeats=2)
    result["pure_bwd_plus_path_ms"] = result["pure_bwd_ms"] + result["path_bwd_ms"]
    result["feature_bytes"] = seed.numel() * seed.element_size()
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(LNET_SCAN_SGS="strict_ro", LNET_SCAN_VFWD="off", LNET_VSCAN_DECOUPLED="off")
    torch.manual_seed(164)
    evict = l2_buffer()
    rows = []
    for size in (56, 28, 14):
        print(f"vscan {size}d", flush=True)
        inputs = make_inputs(BATCH, size, size, MODES, coarse=False, dtype=torch.bfloat16)
        strict = measure(lambda case=inputs: scans.vertical_backward(*case), evict)
        row = {"shape": (BATCH, size, size, MODES), "strict_ro_ms": strict, "chunks": {}}
        rows.append(row)
        for chunk in (*CHUNKS, BATCH):
            row["chunks"][chunk] = chunk_ops(inputs, chunk, evict)
            torch.cuda.empty_cache()
            payload = {"flush_bytes": evict.numel(), "rows": rows}
            args.out.write_text(json.dumps(payload, indent=2) + "\n")
        del inputs


if __name__ == "__main__":
    main()
