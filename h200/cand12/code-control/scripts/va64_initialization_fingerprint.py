"""Fingerprint untrained VA64 initialization without touching active runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-states", action="store_true")
    a = p.parse_args()
    sys.path[:0] = [str(a.code / "src"), str(a.code / "scripts")]
    import numpy as np
    import run_va64_retrain_imagenet1k as runner
    import torch

    torch.set_num_threads(4)
    out = {
        "seed": 0,
        "source_manifest_sha256": hashlib.sha256(
            (a.code / "SOURCE_MANIFEST.json").read_bytes()
        ).hexdigest(),
        "variants": {},
    }
    for v in "BPA":
        random.seed(0)
        np.random.seed(0)  # noqa: NPY002 - reproduce the existing training RNG contract.
        torch.manual_seed(0)
        m = runner.build_model(v)
        state = m.state_dict()
        whole = hashlib.sha256()
        common = hashlib.sha256()
        for name, tensor in state.items():
            header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
            data = (
                tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
            whole.update(header)
            whole.update(data)
            if ".classifier." not in name and not name.endswith((".phase_x", ".phase_y")):
                common.update(header)
                common.update(data)
        out["variants"][v] = {
            "state_sha256": whole.hexdigest(),
            "common_backbone_sha256": common.hexdigest(),
        }
        if a.save_states:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            file = a.output.parent / f"initial-{v}0.pt"
            torch.save(
                {
                    "model": state,
                    "seed": 0,
                    "variant": v,
                    "trained": False,
                    "source_manifest_sha256": out["source_manifest_sha256"],
                },
                file,
            )
            out["variants"][v]["file_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
