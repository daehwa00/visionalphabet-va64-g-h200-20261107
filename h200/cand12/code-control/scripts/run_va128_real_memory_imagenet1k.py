"""Train VA128 with real features/SwiGLU and complex state only inside memory."""

from __future__ import annotations

from run_va_retrain_imagenet1k import arguments, train
from run_va_retrain_imagenet1k import build_model as build_variant

from lnet.real_memory_backbone import RealMemoryBackbone


def build_model(variant: str = "B") -> RealMemoryBackbone:
    model = build_variant(variant, architecture="real-memory")
    if not isinstance(model, RealMemoryBackbone):
        raise TypeError("builder did not produce the real-memory architecture")
    return model


if __name__ == "__main__":
    train(arguments(default_architecture="real-memory"))
