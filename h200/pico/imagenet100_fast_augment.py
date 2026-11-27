# ruff: noqa: ANN001, ANN201, ANN202, ANN204, N812
"""ImageNet-100 training preprocessing for pinned uint8 CUDA prefetch."""

from __future__ import annotations

import math
from itertools import permutations

import torch
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_rgb(path: str) -> Image.Image:
    """Use the JPEG decoder for RGB/grayscale JPEG and PIL for other inputs."""
    with Image.open(path) as header:
        is_jpeg = header.format == "JPEG" and header.mode in {"RGB", "L"}
    if is_jpeg:
        try:
            return F.to_pil_image(decode_jpeg(read_file(path), mode=ImageReadMode.RGB))
        except (RuntimeError, OSError):
            pass
    return pil_loader(path)


def load_rgb_tensor(path: str) -> torch.Tensor:
    # decode_jpeg rejects CMYK and non-JPEG inputs; PIL supplies RGB for both.
    try:
        return decode_jpeg(read_file(path), mode=ImageReadMode.RGB)
    except (RuntimeError, OSError):
        return F.pil_to_tensor(pil_loader(path))


class Uint8TrainingTransform:
    def __init__(self, crop, randaugment):
        self.crop = crop
        self.randaugment = randaugment

    def __call__(self, image):
        image = self.crop(image)
        flipped = bool(torch.rand(1) < 0.5)
        if flipped:
            image = F.hflip(image)
        if self.randaugment is None:
            # RandAugment receives the already flipped image on CUDA.
            return (image if isinstance(image, torch.Tensor) else F.pil_to_tensor(image)), False
        image = self.randaugment(image)
        # Preserve the flip-before-RandAugment ordering, including signed
        # geometric ops, while carrying the final orientation as metadata.
        if flipped:
            image = F.hflip(image)
        return F.pil_to_tensor(image), flipped


def collate_uint8(batch):
    inputs, targets = zip(*batch, strict=True)
    images, flips = zip(*inputs, strict=True)
    return torch.stack(images), torch.tensor(
        list(zip(targets, flips, strict=True)), dtype=torch.long
    )


def _orient(images, packed_targets):
    return torch.where(
        packed_targets[:, 1, None, None, None] != 0,
        images.flip((-1,)),
        images,
    )


def color_jitter(images, factors, orders):
    """Apply per-image brightness, contrast, and saturation in the given order."""
    for operation in orders.unbind(dim=1):
        factor = factors.gather(1, operation[:, None])[:, :, None, None]
        red, green, blue = images.unbind(dim=1)
        gray = (0.2989 * red + 0.587 * green + 0.114 * blue).unsqueeze(dim=1)
        operation_mask = operation[:, None, None, None]
        background = torch.where(operation_mask == 1, gray.mean(dim=(2, 3), keepdim=True), gray)
        background = torch.where(operation_mask == 0, 0.0, background)
        images = (factor * images + (1.0 - factor) * background).clamp(0.0, 1.0)
    return images


@torch.compile(fullgraph=True, dynamic=False)
def _jitter_and_normalize(images, packed_targets, factors, orders, mean, std, channels_last):
    images = _orient(images, packed_targets).float() / 255.0
    images = (color_jitter(images, factors, orders) - mean) / std
    return images.contiguous(memory_format=torch.channels_last) if channels_last else images


@torch.compile(fullgraph=True, dynamic=False)
def _augment(
    images, packed_targets, randoms, axis, mean, std, channels_last, random_erasing_prob=0.25
):
    images = (_orient(images, packed_targets).float() / 255.0 - mean) / std
    if random_erasing_prob == 0.0:
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return images, torch.zeros(len(images), dtype=torch.bool, device=images.device)

    height, width = images.shape[-2:]
    area = height * width * (0.02 + randoms[:, 1] * 0.31)
    log_ratio = math.log(0.3) + randoms[:, 2] * (math.log(3.3) - math.log(0.3))
    ratio = log_ratio.exp()
    erase_h = (area * ratio).sqrt().round().long()
    erase_w = (area / ratio).sqrt().round().long()
    valid = (erase_h < height) & (erase_w < width)
    first = valid.int().argmax(dim=1)[:, None]

    def select(values):
        return values.gather(1, first).reshape(-1, 1, 1)

    top = select((randoms[:, 3] * (height - erase_h + 1)).long())
    left = select((randoms[:, 4] * (width - erase_w + 1)).long())
    enabled = (randoms[:, 0, 0] < random_erasing_prob) & valid.any(dim=1)
    yy = axis[None, :, None] - top
    xx = axis[None, None, :] - left
    erased = (
        enabled[:, None, None]
        & (yy >= 0)
        & (yy < select(erase_h))
        & (xx >= 0)
        & (xx < select(erase_w))
    )
    images = images.masked_fill(erased[:, None], 0)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, enabled


class CudaTrainingAugment:
    def __init__(
        self, device, *, channels_last, random_erasing_prob=0.25, color_jitter_strength=0.0
    ):
        self.random_erasing_prob = random_erasing_prob
        self.color_jitter_strength = color_jitter_strength
        if color_jitter_strength and random_erasing_prob:
            raise ValueError("Fast ColorJitter requires RandomErasing to be disabled")
        if color_jitter_strength:
            self.orders = torch.tensor(list(permutations(range(3))), device=device)
        self.device = device
        self.channels_last = channels_last
        self.axis = torch.arange(224, device=device)
        self.mean = torch.tensor(MEAN, device=device)[None, :, None, None]
        self.std = torch.tensor(STD, device=device)[None, :, None, None]

    def __call__(self, images, packed_targets):
        if self.color_jitter_strength:
            orders = self.orders[
                torch.randint(len(self.orders), (len(images),), device=self.device)
            ]
            strength = self.color_jitter_strength
            factors = torch.empty((len(images), 3), dtype=torch.float32, device=self.device)
            factors.uniform_(1.0 - strength, 1.0 + strength)
            images = _jitter_and_normalize(
                images,
                packed_targets,
                factors,
                orders,
                self.mean,
                self.std,
                self.channels_last,
            )
        else:
            randoms = (
                torch.rand((len(images), 5, 10), device=self.device)
                if self.random_erasing_prob
                else None
            )
            images, _ = _augment(
                images,
                packed_targets,
                randoms,
                self.axis,
                self.mean,
                self.std,
                self.channels_last,
                self.random_erasing_prob,
            )
        return images, packed_targets[:, 0].contiguous()
