"""Full-model CPU FakeTensor and BF16 eager smoke test for the Wave 34 presets."""

import os

import torch
from run_va_eight_ablations import build_variant
from torch._subclasses.fake_tensor import FakeTensorMode

VARIANT = "17-p1-nr4-1024d-v3"


def smoke(model: torch.nn.Module) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(inputs)[0]
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    gradients = tuple(p.grad for p in model.parameters() if p.grad is not None)
    if logits.shape != (2, 100) or not gradients:
        raise RuntimeError("Full model smoke produced an unexpected output or no gradients")
    return loss, gradients


def main() -> None:
    torch.manual_seed(124)
    expected = {
        "LNET_SCAN_W34A": "modefirst_tc_gs1",
        "LNET_READER_W34B": "full_w34b",
        "LNET_MODEL_W34C": "full_w34c",
    }
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Set all Wave 34 smoke presets")
    model = build_variant(VARIANT).to(memory_format=torch.channels_last).train()
    with FakeTensorMode(allow_non_fake_inputs=True):
        fake_loss, fake_grads = smoke(model)
    model = build_variant(VARIANT).to(memory_format=torch.channels_last).train()
    loss, grads = smoke(model)
    if not bool(loss.isfinite()) or not all(bool(g.isfinite().all()) for g in grads):
        raise RuntimeError("CPU BF16 eager smoke found nonfinite values")
    if len(fake_grads) != len(grads) or fake_loss.shape != loss.shape:
        raise RuntimeError("FakeTensor and eager gradients or loss shapes differ")
    print(f"fake/eager gradient tensors={len(grads)}; BF16 eager loss={loss.item():.6f}")  # noqa: T201


if __name__ == "__main__":
    main()
