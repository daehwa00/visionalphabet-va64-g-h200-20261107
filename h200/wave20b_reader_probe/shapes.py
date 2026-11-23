"""Trace real model reader calls, or inspect their shapes with a CPU stand-in."""

import argparse
import json
from unittest.mock import patch

import torch
from run_va_eight_ablations import build_variant

from lnet import memory_backbone

VARIANTS = (
    "17-p1-nr4-1024d-v3",
    "17-live-complex-rms-s03",
    "17-e2-multiscale-s3s4-live-rms-v2",
)


def calls(variant: str, batch: int, device: str) -> list[dict]:
    model = build_variant(variant).to(device).eval()
    layers = []
    trace = []

    def record(name: str, source: torch.Tensor, modes: int, rank: int) -> None:
        trace.append({"layer": name, "batch": batch, "height": source.shape[1],
                      "width": source.shape[2], "modes": modes, "rank": rank,
                          "dtype": str(source.dtype), "strict": False})

    def packed(
        source: torch.Tensor, spatial: torch.Tensor, *, component_first: bool,
    ) -> torch.Tensor:
        assert component_first  # noqa: S101
        _, modes, rank, _, _ = spatial.shape
        record(layers[-1], source, modes, rank)
        return real_packed(source, spatial, component_first=component_first)

    def stand_in(
        layer: memory_backbone.FactorizedMemoryReader, source: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.nn.functional.linear(source, layer.point_weight.flatten(0, 1))
        record(layers[-1], projected, layer.output_modes, layer.rank)
        return projected.new_zeros((*projected.shape[:-1], 2 * layer.output_modes))

    for name, layer in model.named_modules():
        if isinstance(layer, memory_backbone.FactorizedMemoryReader):
            layer.register_forward_pre_hook(lambda *_args, name=name: layers.append(name))
    real_packed = memory_backbone.real_packed_spatial
    hook = (
        patch.object(memory_backbone, "real_packed_spatial", packed) if device == "cuda"
        else patch.object(memory_backbone.FactorizedMemoryReader, "forward", stand_in)
    )
    with hook, torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        model(torch.randn(batch if device == "cuda" else 1, 3, 224, 224, device=device))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", nargs="+", type=int, default=(128, 1024, 1280, 1536))
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    for variant in args.variant:
        for batch in args.batch:
            print(json.dumps({"variant": variant, "calls": calls(variant, batch, args.device)}))  # noqa: T201


if __name__ == "__main__":
    main()
