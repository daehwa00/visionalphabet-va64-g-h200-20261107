# ruff: noqa: ANN001, ANN202, ANN204, PLR1714, SLF001
"""Batched uint8 torchvision RandAugment with PIL-compatible pixel operations."""

from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl
from PIL import Image
from torchvision.transforms import RandAugment
from torchvision.transforms.autoaugment import _apply_op

SIZE = 224
PIXELS = SIZE * SIZE
N_OPS = 14
TILE = 2048
N_TILES = triton.cdiv(PIXELS, TILE)


@lru_cache(maxsize=4)
def _tables(magnitude):
    spec = RandAugment(magnitude=magnitude)
    space = spec._augmentation_space(spec.num_magnitude_bins, (SIZE, SIZE))
    if len(space) != N_OPS:
        raise ValueError("Unexpected RandAugment operation count")
    index = np.arange(PIXELS, dtype=np.int32).reshape(SIZE, SIZE) + 1
    encoded = Image.fromarray(np.stack((index & 255, index >> 8, index * 0), -1).astype(np.uint8))
    mapping = np.zeros((12, PIXELS), dtype=np.int32)
    levels = np.arange(256, dtype=np.uint8)[None, :]
    base = np.broadcast_to(levels.T, (256, 256))
    source = np.broadcast_to(levels, (256, 256))
    point = np.zeros((28, 256), dtype=np.uint8)
    blend = np.zeros((2, 256, 256), dtype=np.uint8)
    for op, (name, (magnitudes, signed)) in enumerate(space.items()):
        amount = float(magnitudes[magnitude]) if magnitudes.ndim else 0.0
        for sign in range(2):
            level = -amount if signed and sign else amount
            choice = op * 2 + sign
            if op < 6:
                pixels = np.asarray(
                    _apply_op(encoded, name, level, spec.interpolation, spec.fill), dtype=np.int32
                )
                mapping[choice] = (pixels[..., 0] + 256 * pixels[..., 1] - 1).flatten()
            elif op in (6, 10, 11):
                point[choice] = np.asarray(
                    _apply_op(
                        Image.fromarray(levels.copy().repeat(3, axis=0)),
                        name,
                        level,
                        spec.interpolation,
                        spec.fill,
                    )
                )[0]
            elif op == 7:
                blend[sign] = np.asarray(
                    Image.blend(Image.fromarray(base), Image.fromarray(source), 1.0 + level)
                )
    low = np.arange(256)[:, None, None]
    high = np.arange(256)[None, :, None]
    value = np.arange(256)[None, None, :]
    scale = 255.0 / np.maximum(high - low, 1)
    # PIL calculates the offset separately before applying the affine LUT.
    autocontrast = np.where(high > low, np.clip(value * scale - low * scale, 0, 255), value).astype(
        np.uint8
    )
    return mapping, point, blend, autocontrast


@triton.jit
def _luma(x, image, p, valid, pixels: tl.constexpr):
    r = tl.load(x + (image * 3) * pixels + p, valid, 0).to(tl.int32)
    g = tl.load(x + (image * 3 + 1) * pixels + p, valid, 0).to(tl.int32)
    b = tl.load(x + (image * 3 + 2) * pixels + p, valid, 0).to(tl.int32)
    return (19595 * r + 38470 * g + 7471 * b + 32768) >> 16


@triton.jit
def _partial_hist(
    x, choices, partials, pixels: tl.constexpr, tile: tl.constexpr, tiles: tl.constexpr
):
    image = tl.program_id(0)
    c = tl.program_id(1)
    chunk = tl.program_id(2)
    op = tl.load(choices + image) // 2
    if (op == 8 and c == 0) or op >= 12:
        p = chunk * tile + tl.arange(0, tile)
        valid = p < pixels
        if op == 8:
            values = _luma(x, image, p, valid, pixels)
        else:
            values = tl.load(x + (image * 3 + c) * pixels + p, valid, 0).to(tl.int32)
        hist = tl.histogram(values, 256, mask=valid)
        tl.store(partials + ((image * 3 + c) * tiles + chunk) * 256 + tl.arange(0, 256), hist)


@triton.jit
def _build_lut(
    choices,
    partials,
    luts,
    autocontrast,
    pixels: tl.constexpr,
    tiles: tl.constexpr,
    chunks: tl.constexpr,
):
    image = tl.program_id(0)
    c = tl.program_id(1)
    op = tl.load(choices + image) // 2
    if (op == 8 and c == 0) or op >= 12:
        chunk = tl.arange(0, chunks)
        values = tl.arange(0, 256)
        h = tl.load(
            partials + ((image * 3 + c) * tiles + chunk[:, None]) * 256 + values[None, :],
            chunk[:, None] < tiles,
            0,
        )
        hist = tl.sum(h, 0)
        hi = tl.max(tl.where(hist > 0, values, 0), 0)
        if op == 8:
            mean = (tl.sum(hist * values, 0) + pixels // 2) // pixels
            result = tl.full((256,), 0, tl.int32) + mean
        elif op == 12:
            lo = tl.min(tl.where(hist > 0, values, 255), 0)
            result = tl.load(autocontrast + (lo * 256 + hi) * 256 + values).to(tl.int32)
        else:
            last = tl.sum(tl.where(values == hi, hist, 0), 0)
            step = (pixels - last) // 255
            cumulative = tl.cumsum(hist) - hist
            result = tl.where(
                step > 0, tl.minimum((cumulative + step // 2) // tl.maximum(step, 1), 255), values
            )
        tl.store(luts + (image * 3 + c) * 256 + values, result)


@triton.jit
def _apply(
    x,
    out,
    choices,
    mapping,
    point,
    blend,
    luts,
    pixels: tl.constexpr,
    size: tl.constexpr,
    block: tl.constexpr,
):
    image = tl.program_id(0)
    c = tl.program_id(1)
    p = tl.program_id(2) * block + tl.arange(0, block)
    valid = p < pixels
    choice = tl.load(choices + image)
    op = choice // 2
    base = x + (image * 3 + c) * pixels
    if op < 6:
        idx = tl.load(mapping + choice * pixels + p, valid, -1)
        value = tl.load(base + idx, idx >= 0, 0).to(tl.int32)
    else:
        pixel = tl.load(base + p, valid, 0).to(tl.int32)
        if op == 6 or op == 10 or op == 11:
            value = tl.load(point + choice * 256 + pixel).to(tl.int32)
        elif op == 7 or op == 8 or op == 9:
            if op == 7:
                background = _luma(x, image, p, valid, pixels)
            elif op == 8:
                background = tl.full((block,), 0, tl.int32) + tl.load(luts + image * 3 * 256)
            else:
                border = (
                    (p % size == 0) | (p % size == size - 1) | (p < size) | (p >= pixels - size)
                )
                smooth = 5 * pixel
                for delta in tl.static_range(-1, 2):
                    for dx in tl.static_range(-1, 2):
                        if delta != 0 or dx != 0:
                            smooth += tl.load(base + p + delta * size + dx, valid & ~border, 0).to(
                                tl.int32
                            )
                background = tl.where(border, pixel, (smooth + 6) // 13)
            value = tl.load(blend + ((choice % 2) * 256 + background) * 256 + pixel).to(tl.int32)
        else:
            value = tl.load(luts + (image * 3 + c) * 256 + pixel)
    tl.store(out + (image * 3 + c) * pixels + p, value, valid)


class GpuRandAugment:
    def __init__(self, device, *, magnitude=5):
        self.tables = tuple(torch.from_numpy(table).to(device) for table in _tables(magnitude))

    def __call__(self, images, choices=None):
        if (
            images.dtype != torch.uint8
            or images.shape[1:] != (3, SIZE, SIZE)
            or not images.is_contiguous()
        ):
            raise ValueError("RandAugment requires contiguous NCHW uint8 RGB crops of size 224")
        batch = images.shape[0]
        if choices is None:
            choices = torch.randint(N_OPS * 2, (2, batch), device=images.device)
        mapping, point, blend, autocontrast = self.tables
        partials = torch.empty((batch, 3, N_TILES, 256), dtype=torch.int32, device=images.device)
        luts = torch.empty((batch, 3, 256), dtype=torch.int32, device=images.device)
        for selected in choices.unbind(0):
            _partial_hist[(batch, 3, N_TILES)](images, selected, partials, PIXELS, TILE, N_TILES)
            _build_lut[(batch, 3)](
                selected,
                partials,
                luts,
                autocontrast,
                PIXELS,
                N_TILES,
                triton.next_power_of_2(N_TILES),
            )
            out = torch.empty_like(images)
            _apply[(batch, 3, triton.cdiv(PIXELS, 512))](
                images, out, selected, mapping, point, blend, luts, PIXELS, SIZE, 512
            )
            images = out
        return images
