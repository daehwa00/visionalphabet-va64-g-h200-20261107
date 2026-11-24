"""Check full-model fake autograd and CPU BF16 eager gradients."""

import torch
from run_va_eight_ablations import build_variant
from torch._subclasses.fake_tensor import FakeTensorMode

VARIANT = "17-p1-nr4-1024d-v3"


def step(model: torch.nn.Module) -> tuple[torch.Tensor, list[torch.Tensor]]:
    image = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    labels = torch.arange(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(image)[0], labels)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    return loss, gradients


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(501)
    model = build_variant(VARIANT)
    with FakeTensorMode(allow_non_fake_inputs=True):
        _, gradients = step(model)
        if not gradients:
            raise RuntimeError("FakeTensor backward produced no gradients")
    loss, gradients = step(build_variant(VARIANT))
    if not gradients or not torch.isfinite(loss) or not all(g.isfinite().all() for g in gradients):
        raise RuntimeError("CPU BF16 eager loss or gradients failed the finite check")
    print(f"CPU BF16 loss={loss.item():.6f}, finite gradients={len(gradients)}")  # noqa: T201


if __name__ == "__main__":
    main()
