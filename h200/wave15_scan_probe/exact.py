# ruff: noqa: ANN001, ANN201, E501, FBT003, SLF001, T201
"""Compare unrounded tensor digests against a separate Wave 14A process."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from bench import ARMS, CASES, TRIAL_ENABLED, sample

from lnet import general_cuda_scans as scans


def digest(value):
    result = hashlib.sha256()
    for chunk in value.split(16, dim=0):
        result.update(memoryview(chunk.contiguous().view(torch.uint8).cpu().numpy()))
    return result.hexdigest()


def canonical(value, layout):
    if layout == 1:
        return value.permute(4, 0, 1, 2, 3)
    if layout == 2:
        return value.permute(4, 0, 2, 1, 3)
    return value


def probe(shape, batch, arms):
    h, w, m, coarse = shape
    b = batch if h <= 56 else 4
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    dtype = torch.bfloat16
    source = [sample((b, h, w, m), dtype, 0.4) for _ in range(2)]
    px, py = [[sample((m,), torch.float32, 0.12) for _ in range(4)] for _ in range(2)]
    aux = [torch.ones((2, size, m), device="cuda") for size in (w, h)]
    path = [
        sample(dims, torch.float32, 0.17) for dims in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    dy = tuple(sample((b, oh, ow, m), dtype, 0.1) for _ in range(2))
    results = {}
    for arm in arms:
        layout, scratch, single, drop_source = ARMS[arm]
        h_options = {"state_layout": layout} if TRIAL_ENABLED else {}
        v_options = {
            **h_options, "local_scratch": scratch, "single_store": single,
            "drop_source": drop_source,
        } if TRIAL_ENABLED else {}
        states, _ = scans._hscan(px, *source, **h_options)
        horizontal_state = [digest(value) for value in canonical(states, layout)]
        out = scans._vscan(py, states, aux, path, coarse, 1e-6, True, **v_options)
        dv = scans._vscan(py, states, aux, path, coarse, 1e-6, False, dy, compact=coarse, **v_options)
        dx, poles = scans._hscan(px, *source, states, dv[4], compact=coarse, **h_options)
        results[arm] = {
            "states": horizontal_state,
            "output": [digest(value) for value in out],
            "gs": [digest(value) for value in canonical(dv[4], layout)],
            "dx": digest(dx),
            "parameters": [digest(value) for value in (*poles, *dv[:4], *dv[5:])],
        }
        del states, out, dv, dx, poles
    return {"shape": [b, h, w, m, coarse], "dtype": str(dtype), "digests": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=["baseline"])
    args = parser.parse_args()
    torch.manual_seed(731)
    with args.out.open("w") as out, torch.no_grad():
        for shape in CASES:
            result = probe(shape, args.batch, args.arms)
            print(json.dumps(result), file=out, flush=True)
            print(result["shape"], flush=True)


if __name__ == "__main__":
    main()
