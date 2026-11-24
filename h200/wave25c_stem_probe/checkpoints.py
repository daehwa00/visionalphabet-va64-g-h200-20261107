"""Create and reload reference or candidate state dictionaries on CPU."""

# ruff: noqa: T201
import argparse
from pathlib import Path

import torch
from run_va_eight_ablations import build_variant

VARIANTS = ("17-p1-nr4-1024d-v3", "17-live-complex-rms-s03", "17-e2-multiscale-s3s4-live-rms-v2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        model = build_variant(variant)
        state = model.state_dict()
        if args.reference:
            incoming = torch.load(args.reference / f"{variant}.pt", weights_only=True)
            if tuple(state) != tuple(incoming):
                raise ValueError(f"{variant}: state dictionary keys differ")
            model.load_state_dict(incoming, strict=True)
            print(variant, len(state), "strict reload passed", flush=True)
        torch.save(model.state_dict(), args.output / f"{variant}.pt")


if __name__ == "__main__":
    main()
